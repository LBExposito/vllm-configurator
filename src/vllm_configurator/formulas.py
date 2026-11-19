"""Pure sizing formulas (docs/vllm-config-knowledge-base.md, section 3).

Notation: B = max_num_batched_tokens, S = active sequences (<= max_num_seqs),
k = num_speculative_tokens, a = mean acceptance length (tokens emitted per step
and per sequence), d = decode tokens charged to the budget per sequence.
"""

from __future__ import annotations

import math


def required_ratio(budget: int, seqs: int, decode_tokens_per_seq: float, acceptance_length: float) -> float:
    """§3.1: ISL_eff/OSL ratio needed to fill the token budget in steady state.

    About S·a/OSL requests finish per step and are replaced by as many new ones,
    each bringing ISL_eff prefill tokens: ISL_eff/OSL >= (B - S·d) / (S·a).
    """
    if seqs <= 0 or acceptance_length <= 0:
        raise ValueError("seqs et acceptance_length doivent être > 0")
    return (budget - seqs * decode_tokens_per_seq) / (seqs * acceptance_length)


def effective_ratio(isl: float, prefix_cache_hit_rate: float, osl: float) -> float:
    """ISL_eff/OSL: only the part of the prompt missing from the prefix cache is prefilled."""
    if not 0.0 <= prefix_cache_hit_rate <= 1.0:
        raise ValueError("prefix_cache_hit_rate doit être dans [0, 1]")
    if osl <= 0:
        raise ValueError("osl doit être > 0")
    return isl * (1.0 - prefix_cache_hit_rate) / osl


def decode_tokens_per_seq(num_speculative_tokens: int, extra_draft_slots: int, verification_in_budget: bool = True) -> int:
    """d in §3.1: budget tokens charged per decoding sequence and per step.

    Without spec decoding d = 1. With spec decoding the k draft tokens are verified
    by the target model and charged to the budget (vLLM >= 0.31 scheduler), plus
    any extra drafting slots the method reserves (max_num_new_slots_for_drafting).
    """
    if num_speculative_tokens == 0:
        return 1
    verified = num_speculative_tokens if verification_in_budget else 0
    return 1 + verified + extra_draft_slots


def eval_slot_expression(expr: str, k: int) -> int:
    """Evaluate a draft-slot expression from the vLLM catalog: '0', '1', 'k', 'k-1'."""
    expr = expr.replace(" ", "")
    if expr == "k":
        return k
    if expr == "k-1":
        return k - 1
    if expr.isdigit():
        return int(expr)
    raise ValueError(f"expression de slots inconnue : {expr!r}")


def mean_from_percentiles(p50: float, p95: float | None = None, p99: float | None = None) -> float:
    """Mean of a length distribution from its percentiles (lognormal fit, with a heavy-tail correction).

    The lognormal is fitted on P50/P95 (or P50/P99). When the observed P99 exceeds the fitted one, at least 1 % of
    the requests are that much longer: the mean gets 0.01 × (P99 − P99_fitted) on top. Without P95 nor P99: P50.
    """
    if p95 and p95 > p50:
        sigma = math.log(p95 / p50) / 1.645
    elif p99 and p99 > p50:
        sigma = math.log(p99 / p50) / 2.326
    else:
        return float(p50)
    mean = p50 * math.exp(sigma**2 / 2)
    if p95 and p99:
        fitted_p99 = p50 * math.exp(2.326 * sigma)
        if p99 > fitted_p99:
            mean += 0.01 * (p99 - fitted_p99)
    return mean


def kv_concurrency(kv_tokens: float, context_tokens: float) -> float:
    """§3.2: number of contexts of a given length that fit in the KV cache."""
    if context_tokens <= 0:
        raise ValueError("context_tokens doit être > 0")
    return kv_tokens / context_tokens


def cudagraph_capture_size(num_speculative_tokens: int, max_num_seqs: int) -> int:
    """§3.3: max_cudagraph_capture_size = (k + 1) × max_num_seqs."""
    return (num_speculative_tokens + 1) * max_num_seqs


def clamp_capture_size(size: int, budget: int, multiple: int) -> int:
    """vLLM caps the capture size at max_num_batched_tokens and rounds it down to a multiple of k + 1."""
    capped = min(size, budget)
    return max(multiple, capped // multiple * multiple)


def verified_tokens_per_step(seqs: int, num_speculative_tokens: int) -> int:
    """§3.4: tokens verified by the target model in a pure decode step, S × (k + 1)."""
    return seqs * (num_speculative_tokens + 1)


def lru_capacity_per_server(kv_tokens: int, block_size_tokens: int) -> int:
    """§3.5 (v2, llm-d EPP): LRU entries needed to track the whole KV cache of one pod."""
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens doit être > 0")
    return kv_tokens // block_size_tokens


def tokens_per_expert(step_tokens: float, top_k: int, n_experts: int) -> float:
    """§3.6: average tokens routed to each expert of an MoE layer in one step."""
    if n_experts <= 0:
        raise ValueError("n_experts doit être > 0")
    return step_tokens * top_k / n_experts


def max_concurrency_for_length(kv_tokens: int, request_tokens: int) -> float:
    """The « Maximum concurrency for N tokens per request » line of the vLLM startup log."""
    return kv_tokens / request_tokens


def long_prefill_effective_threshold(
    threshold: int,
    budget: int,
    num_running: int,
    num_waiting: int,
    *,
    single_request_bypass: bool,
    adaptive: bool,
) -> int:
    """Effective per-request prefill cap of the vLLM V1 scheduler (0 = no cap).

    Mirrors vllm/v1/core/sched/scheduler.py (>= 0.31): no cap when a single request
    is present (#57951); with --long-prefill-token-threshold-adaptive the cap is
    floored at budget // (running + waiting) (#58459). Decoding requests count in
    the divisor because they are in `running`.
    """
    eligible = num_running + num_waiting
    if threshold <= 0 or (single_request_bypass and eligible <= 1):
        return 0
    if adaptive and eligible > 0:
        return max(threshold, budget // eligible)
    return threshold


def dspark_active_seq_cap(budget: int, num_lookahead_tokens: int) -> int:
    """vLLM >= 0.33 caps max_num_active_seqs at B // num_lookahead_tokens for DSpark/DFlash (#59468)."""
    if num_lookahead_tokens <= 0:
        raise ValueError("num_lookahead_tokens doit être > 0")
    return budget // num_lookahead_tokens


def kv_tokens_estimate(
    *,
    hbm_bytes_per_gpu: float,
    gpu_memory_utilization: float,
    weights_bytes: float,
    tensor_parallel_size: int,
    reserve_bytes_per_gpu: float,
    bytes_per_token: float,
    replicated_across_tp: bool,
    num_kv_heads: int | None,
) -> int:
    """Upper-bound estimate of KV tokens per instance when no startup log is available.

    Per-GPU free memory = HBM × utilization − weights / TP − reserve. With MLA/MQA the
    per-token KV is replicated on every TP rank; with GQA it is split over min(TP, kv_heads).
    Token capacity of the instance equals the per-GPU capacity (each rank holds a shard
    of every token).
    """
    free = hbm_bytes_per_gpu * gpu_memory_utilization - weights_bytes / tensor_parallel_size - reserve_bytes_per_gpu
    if free <= 0:
        return 0
    if replicated_across_tp:
        per_gpu = bytes_per_token
    else:
        shards = min(tensor_parallel_size, num_kv_heads or tensor_parallel_size)
        per_gpu = bytes_per_token / shards
    return math.floor(free / per_gpu)


# --- Distributions used by the SLO analysis ------------------------------------

def normal_ppf(q: float) -> float:
    """Inverse of the standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q doit être dans ]0, 1[")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    if q < 0.02425:
        r = math.sqrt(-2 * math.log(q))
        return ((((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
                / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1))
    if q > 1 - 0.02425:
        return -normal_ppf(1 - q)
    s = q - 0.5
    r = s * s
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * s
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def lognormal_quantile_fn(p50: float, p95: float | None = None, p99: float | None = None, cap: float | None = None):
    """Quantile function of a length distribution, fitted like mean_from_percentiles.

    Lognormal on P50/P95 (or P50/P99). When the observed P99 exceeds the fitted one, the tail above the 99th
    percentile is stretched so that Q(0.99) = P99. Values are capped at `cap` (the served context length)."""
    if p95 and p95 > p50:
        sigma = math.log(p95 / p50) / 1.645
    elif p99 and p99 > p50:
        sigma = math.log(p99 / p50) / 2.326
    else:
        sigma = 0.0
    stretch = 1.0
    if p95 and p99 and sigma:
        fitted_p99 = p50 * math.exp(2.326 * sigma)
        if p99 > fitted_p99:
            stretch = p99 / fitted_p99

    def quantile(q: float) -> float:
        value = p50 * math.exp(sigma * normal_ppf(q))
        if q >= 0.99 and stretch > 1:
            value *= stretch
        return min(value, cap) if cap else value

    return quantile


# Bornes des tranches de probabilité de la grille : fines dans la queue, où se joue le p95.
GRID_EDGES = [i / 40 for i in range(37)] + [0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]
_F_EDGES = [i / 12 for i in range(11)] + [0.95, 0.99, 1.0]
# Tranches de la distribution réduite utilisée point par point (TTFT, ITL) : plus larges sous le 90e centile.
# Pas de borne à 0,95 : le p95 tombe au milieu d'une tranche et s'interpole (quantile_interpolated).
COLLAPSE_EDGES = [i / 20 for i in range(19)] + [0.92, 0.94, 0.955, 0.965, 0.975, 0.985, 0.99, 0.995, 0.999, 1.0]


def _bins(edges: list[float]) -> list[tuple[float, float]]:
    """(probability, representative quantile) of each slice [edges[i], edges[i + 1]]."""
    return [(hi - lo, (lo + hi) / 2) for lo, hi in zip(edges, edges[1:])]


def uncached_distribution(isl_quantile, hit: float, cold_fraction: float, warm_sigma: float,
                          shared_prefix: float = 0.0) -> tuple[list[tuple[float, float, float]], float]:
    """Distribution of the prompt tokens to prefill (missing from the prefix cache), as (probability, U, ISL) points.

    A request is cold with probability `cold_fraction` (new session past the shared prefix, eviction, misrouting):
    U = ISL − shared prefix. Otherwise it is warm: U = min(ISL, X), X lognormal of dispersion `warm_sigma` and
    independent of ISL. What a turn adds (tool outputs, user message) does not grow with the history; with a low hit
    rate the cap brings U back to ~ISL. The median of X is solved so that the token-weighted hit rate equals `hit`.
    If the cold requests alone exceed the uncached share, X = 0 and the effective hit rate is lower than `hit`. With
    hit = 0 every prompt is prefilled in full. Returns (points sorted by U, effective hit rate).
    """
    if not 0.0 <= hit <= 1.0 or not 0.0 <= cold_fraction <= 1.0:
        raise ValueError("hit et cold_fraction doivent être dans [0, 1]")
    isl_bins = [(prob, isl_quantile(q)) for prob, q in _bins(GRID_EDGES)]
    mean_isl = sum(prob * isl for prob, isl in isl_bins)
    if hit <= 0.0:
        return sorted(((prob, isl, isl) for prob, isl in isl_bins), key=lambda t: t[1]), 0.0
    cold = ([(prob * cold_fraction, max(isl - shared_prefix, 1.0), isl) for prob, isl in isl_bins]
            if cold_fraction else [])
    warm_target = (1.0 - hit) * mean_isl - sum(p * u for p, u, _ in cold)
    warm_share = 1.0 - cold_fraction
    f_bins = [(prob, normal_ppf(q)) for prob, q in _bins(_F_EDGES)]

    def warm_points(median: float) -> list[tuple[float, float, float]]:
        return [(pi * pf * warm_share, min(isl, median * math.exp(warm_sigma * z)), isl)
                for pi, isl in isl_bins for pf, z in f_bins]

    def mean(points) -> float:
        return sum(p * u for p, u, _ in points)

    if warm_share <= 0:
        warm = []
    elif warm_target <= 0:
        warm = [(prob * warm_share, 0.0, isl) for prob, isl in isl_bins]
    else:
        lo, hi = 0.0, 1.0
        while mean(warm_points(hi)) < warm_target and hi < 1e12:
            hi *= 2
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean(warm_points(mid)) < warm_target else (lo, mid)
        warm = warm_points((lo + hi) / 2)
    points = sorted(cold + warm, key=lambda t: t[1])
    return points, (1.0 - mean(points) / mean_isl) if mean_isl else hit


def collapse(points: list[tuple[float, float, float]], edges: list[float] | None = None) -> list[tuple[float, float, float]]:
    """Merge (probability, U, ISL) points sorted by U into probability slices (fine in the tail), keeping the
    mean U and the mean ISL of each slice."""
    edges = edges or COLLAPSE_EDGES
    out: list[tuple[float, float, float]] = []
    acc, i = 0.0, 0
    for hi in edges[1:]:
        prob = su = si = 0.0
        while i < len(points) and acc + points[i][0] <= hi + 1e-9:
            p, u, isl = points[i]
            prob, su, si, acc, i = prob + p, su + p * u, si + p * isl, acc + p, i + 1
        if prob > 0:
            out.append((prob, su / prob, si / prob))
    if i < len(points):  # reste dû aux arrondis
        rest = points[i:]
        prob = sum(p for p, _, _ in rest)
        if prob > 0:
            out.append((prob, sum(p * u for p, u, _ in rest) / prob, sum(p * isl for p, _, isl in rest) / prob))
    return out


def quantile_interpolated(points: list[tuple[float, float]], q: float) -> float:
    """q-quantile of a discrete distribution of (probability, value) slices sorted by value, each value standing at
    the middle of its slice: linear interpolation between the middles of the two neighbouring slices."""
    if not points:
        return 0.0
    acc, prev = 0.0, None
    for p, v in points:
        mid = acc + p / 2
        if q <= mid:
            if prev is None:
                return v
            pm, pv = prev
            return pv + (v - pv) * (q - pm) / (mid - pm) if mid > pm else v
        prev, acc = (mid, v), acc + p
    return points[-1][1]


def quantile_of_points(points: list[tuple[float, float, float]], q: float) -> float:
    """q-quantile of U over (probability, U, ISL) points sorted by U."""
    return quantile_interpolated([(p, u) for p, u, _ in points], q)


def tokens_per_step(alpha: float, k: int) -> float:
    """Tokens emitted per step and per sequence when each of the k drafts is accepted with probability alpha
    (until the first rejection): 1 + alpha + … + alpha^k."""
    if k <= 0:
        return 1.0
    if alpha >= 1.0:
        return k + 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def acceptance_quantile(mean_length: float, k: int, q: float, concentration: float) -> float:
    """q-quantile over the requests of the acceptance length a, from its mean.

    The per-draft acceptance rate of a request follows a Beta whose mean is solved from the mean a and whose
    concentration (alpha + beta) is given."""
    if k <= 0 or mean_length <= 1.0:
        return 1.0
    mean_length = min(mean_length, k + 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tokens_per_step(mid, k) < mean_length else (lo, mid)
    alpha = min(max((lo + hi) / 2, 1e-3), 0.999)
    return tokens_per_step(beta_ppf(q, alpha * concentration, (1 - alpha) * concentration), k)


def beta_ppf(q: float, a: float, b: float, steps: int = 4000) -> float:
    """q-quantile of a Beta(a, b) law, by numerical integration of its density (midpoint rule)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q doit être dans ]0, 1[")
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    width = 1.0 / steps
    xs = [(i + 0.5) * width for i in range(steps)]
    dens = [math.exp(log_norm + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x)) * width for x in xs]
    total = sum(dens)
    acc = 0.0
    for x, d in zip(xs, dens):
        acc += d / total
        if acc >= q:
            return x
    return xs[-1]
