"""First-order latency model of one vLLM scheduler step, and the queueing formulas of the SLO analysis.

A step processes T = decode tokens (k + 1 verified per running sequence) + prompt tokens (chunked prefill). The
kernels of a step run one after the other, so its duration is the SUM of the operators, each bounded by its own
roofline max(bytes / bandwidth, FLOPs / throughput). One global max(memory, compute) would let the prefill compute
hide under the weight reads, which no real step does.

- dense (non-routed) weights: read once per step, 2 FLOPs per parameter and token;
- routed experts (MoE): only the experts hit by at least one token are read (uniform routing);
- attention: decode reads the KV of every sequence; FLOPs grow with the context, with a full-context part (dense
  attention, or the indexer of a sparse attention) and a part over the selected tokens (sparse attention);
- TP all-reduce: 2 per layer, T × hidden × 2 bytes, over NVLink (ring: 2 (TP − 1) / TP of the data per GPU);
- fixed cost per step (kernel launches, all-reduce latency, scheduler), higher on mixed steps that run with
  piecewise CUDA graphs; a cost per running sequence (sampling, rejection sampling, bookkeeping);
- drafter pass of speculative decoding, as a fraction of the target decode step.

All quantities are per GPU: weights, FLOPs and the query heads are split over the TP ranks (the KV per GPU is
passed in as is, already taking replication into account).

The decode part (step without prompt tokens) and the prefill part (what the prompt tokens add) are scaled by
separate factors: a measured ITL calibrates the first, a measured TTFT the second.

This is an order of magnitude, not a prediction: the analysis gives every derived value as a range and lets
measurements recalibrate it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class PerfModel:
    tp: int
    hbm_bytes_per_s: float          # effective, per GPU
    flops_per_s: float              # effective, per GPU
    overhead_s: float               # fixed per step (pure decode, full CUDA graphs)
    active_params_decode: float     # per output token
    active_params_prefill: float    # per prompt token
    dense_params: float             # non-routed weights read at every step
    dense_bytes_per_param: float
    routed_params: float            # all routed experts (0 for a dense model)
    routed_active_params: float     # routed parameters used by one token (top-k experts)
    routed_bytes_per_param: float
    experts_per_tok: int
    n_experts: int
    kv_bytes_per_token_per_gpu: float | None
    # Attention FLOPs per query token: attn_full_flops × context + attn_sparse_flops × min(context, attn_sparse_tokens).
    attn_full_flops: float = 0.0
    attn_sparse_flops: float = 0.0
    attn_sparse_tokens: float = 0.0
    hidden_size: int = 0
    num_layers: int = 0
    allreduce_bytes_per_s: float | None = None   # effective, per GPU, one direction
    mixed_overhead_factor: float = 1.0
    per_seq_overhead_s: float = 0.0
    drafter_fraction: float = 0.0
    decode_factor: float = 1.0
    prefill_factor: float = 1.0

    def scaled(self, decode: float, prefill: float) -> "PerfModel":
        """The same model with its decode and prefill parts multiplied by these factors."""
        return replace(self, decode_factor=self.decode_factor * decode, prefill_factor=self.prefill_factor * prefill)

    def calibrated(self, decode: float, prefill: float) -> "PerfModel":
        return replace(self, decode_factor=decode, prefill_factor=prefill)

    def attention_flops_per_query(self, context: float) -> float:
        return self.attn_full_flops * context + self.attn_sparse_flops * min(context, self.attn_sparse_tokens)


@dataclass(frozen=True)
class StepTime:
    seconds: float
    memory_s: float
    compute_s: float
    overhead_s: float
    weights_bytes: float
    kv_bytes: float
    comm_s: float = 0.0
    drafter_s: float = 0.0
    decode_s: float = 0.0           # part of `seconds` that a step without prompt tokens would take
    ops: dict[str, float] = field(default_factory=dict)

    @property
    def bound(self) -> str:
        """« memory » or « compute »: the roofline term that dominates the operators of the step."""
        return "memory" if self.memory_s >= self.compute_s else "compute"

    @property
    def ms(self) -> float:
        return self.seconds * 1e3


def experts_hit_fraction(tokens: float, experts_per_tok: int, n_experts: int) -> float:
    """Expected fraction of routed experts hit by at least one of `tokens` tokens (uniform routing)."""
    if n_experts <= 0 or tokens <= 0:
        return 0.0
    return 1.0 - (1.0 - experts_per_tok / n_experts) ** tokens


def _raw_step(m: PerfModel, decode_seqs: int, tokens_per_seq: int, context_tokens: float, prefill_tokens: float,
              prefill_context: float) -> dict[str, tuple[float, float]]:
    """(memory s, compute s) of each operator, plus fixed costs as (s, 0), unscaled."""
    decode_tokens = decode_seqs * tokens_per_seq
    tokens = decode_tokens + prefill_tokens
    ops: dict[str, tuple[float, float]] = {}
    dense_decode = max(m.active_params_decode - m.routed_active_params, 0.0)
    dense_prefill = max(m.active_params_prefill - m.routed_active_params, 0.0)
    ops["dense"] = (m.dense_params * m.dense_bytes_per_param / m.tp / m.hbm_bytes_per_s,
                    2.0 * (dense_decode * decode_tokens + dense_prefill * prefill_tokens) / m.tp / m.flops_per_s)
    if m.routed_params:
        hit = experts_hit_fraction(tokens, m.experts_per_tok, m.n_experts)
        ops["experts"] = (m.routed_params * hit * m.routed_bytes_per_param / m.tp / m.hbm_bytes_per_s,
                          2.0 * m.routed_active_params * tokens / m.tp / m.flops_per_s)
    kv = decode_seqs * context_tokens * (m.kv_bytes_per_token_per_gpu or 0.0)
    attn_flops = (decode_tokens * m.attention_flops_per_query(context_tokens)
                  + prefill_tokens * m.attention_flops_per_query(prefill_context))
    ops["attention"] = (kv / m.hbm_bytes_per_s, attn_flops / m.tp / m.flops_per_s)
    if m.tp > 1 and m.allreduce_bytes_per_s:
        volume = 2 * m.num_layers * tokens * m.hidden_size * 2.0 * 2 * (m.tp - 1) / m.tp
        ops["allreduce"] = (volume / m.allreduce_bytes_per_s, 0.0)
    fixed = m.overhead_s * (m.mixed_overhead_factor if prefill_tokens > 0 else 1.0)
    ops["fixed"] = (fixed + m.per_seq_overhead_s * decode_seqs, 0.0)
    return ops


def _total(ops: dict[str, tuple[float, float]]) -> float:
    return sum(max(mem, comp) for mem, comp in ops.values())


def step_time(m: PerfModel, decode_seqs: int, tokens_per_seq: int, context_tokens: float, prefill_tokens: float,
              prefill_context: float = 0.0) -> StepTime:
    """Duration of one step with `decode_seqs` sequences in decode (each verifying `tokens_per_seq` tokens and reading
    `context_tokens` of KV) and `prefill_tokens` prompt tokens whose mean position in their prompt is `prefill_context`."""
    full = _raw_step(m, decode_seqs, tokens_per_seq, context_tokens, prefill_tokens, prefill_context)
    base = full if prefill_tokens <= 0 else _raw_step(m, decode_seqs, tokens_per_seq, context_tokens, 0, 0.0)
    drafter = m.drafter_fraction * _total(base) if decode_seqs and tokens_per_seq > 1 else 0.0
    decode_raw = _total(base) + drafter
    prefill_raw = _total(full) - _total(base)
    seconds = decode_raw * m.decode_factor + prefill_raw * m.prefill_factor
    raw = decode_raw + prefill_raw
    scale = seconds / raw if raw > 0 else 1.0
    weights = (m.dense_params * m.dense_bytes_per_param
               + m.routed_params * experts_hit_fraction(decode_seqs * tokens_per_seq + prefill_tokens, m.experts_per_tok,
                                                        m.n_experts) * m.routed_bytes_per_param) / m.tp
    memory = sum(mem for key, (mem, _) in full.items() if key in ("dense", "experts", "attention"))
    compute = sum(comp for _, comp in full.values())
    return StepTime(
        seconds=seconds, memory_s=memory * scale, compute_s=compute * scale, overhead_s=full["fixed"][0] * scale,
        weights_bytes=weights, kv_bytes=decode_seqs * context_tokens * (m.kv_bytes_per_token_per_gpu or 0.0),
        comm_s=full.get("allreduce", (0.0, 0.0))[0] * scale, drafter_s=drafter * scale,
        decode_s=decode_raw * m.decode_factor,
        ops={key: max(mem, comp) * scale for key, (mem, comp) in full.items()} | ({"drafter": drafter * scale} if drafter else {}),
    )


def prefill_attention_seconds(m: PerfModel, tokens: float, context: float) -> float:
    """Attention compute of `tokens` prompt tokens at mean position `context`, scaled like the prefill part."""
    return tokens * m.attention_flops_per_query(context) / m.tp / m.flops_per_s * m.prefill_factor


# --- Queueing -----------------------------------------------------------------

def erlang_c(servers: int, offered: float) -> float:
    """Probability of waiting in an M/M/c queue with `offered` = λ / μ erlangs (Erlang C)."""
    if servers <= 0:
        raise ValueError("servers doit être > 0")
    if offered <= 0:
        return 0.0
    if offered >= servers:
        return 1.0
    b = 1.0
    for n in range(1, servers + 1):
        b = offered * b / (n + offered * b)
    rho = offered / servers
    return b / (1.0 - rho + rho * b)


def mgc_wait(rho: float, servers: int, mean_service_s: float, scv: float) -> tuple[float, float]:
    """M/G/c wait before service (Allen–Cunneen, Poisson arrivals): (P(wait), mean wait given a wait).

    E[Wq] ≈ C(c, cρ) × E[S] / (c (1 − ρ)) × (1 + SCV) / 2, SCV = squared coefficient of variation of the service
    time. The conditional wait is taken exponential, which gives the tail used for the TTFT percentile."""
    if rho >= 1.0:
        return 1.0, math.inf
    if rho <= 0 or mean_service_s <= 0:
        return 0.0, 0.0
    p_wait = erlang_c(servers, rho * servers)
    if p_wait <= 0:
        return 0.0, 0.0
    mean_wait = p_wait * mean_service_s / (servers * (1.0 - rho)) * (1.0 + scv) / 2.0
    return p_wait, mean_wait / p_wait


def quantile_of_sum(q: float, points: list[tuple[float, float]], p_wait: float, wait_scale_s: float) -> float:
    """q-quantile of X + W, X given as (probability, value) slices and W = 0 with probability 1 − p_wait, else
    exponential of mean `wait_scale_s`.

    X is read as the piecewise-linear CDF through the middles of its slices (as formulas.quantile_interpolated), so
    that the result does not jump from one slice to the next. Bisection on P(X + W > x) = 1 − q, with the waiting
    term P(X ≤ x) × P(W > x − X) summed over the slices below x in O(log n) (cumulated exponential sums)."""
    from bisect import bisect_right

    from .formulas import quantile_interpolated

    if not points:
        return 0.0
    if math.isinf(wait_scale_s):
        return math.inf
    pts = sorted(points, key=lambda t: t[1])
    if p_wait <= 0 or wait_scale_s <= 0:
        return quantile_interpolated(pts, q)
    values = [v for _, v in pts]
    mids, acc = [], 0.0
    for prob, _ in pts:
        mids.append(acc + prob / 2)
        acc += prob
    # cum[j] = Σ_{i ≤ j} p_i exp((v_i − v_j) / θ), without overflow (exponents ≤ 0).
    cum, prev = [], None
    for prob, value in pts:
        cum.append(prob if prev is None else cum[-1] * math.exp((prev - value) / wait_scale_s) + prob)
        prev = value

    def survival(x: float) -> float:
        if x <= values[0]:
            return 1.0 if x < values[0] else 1.0 - mids[0]
        if x >= values[-1]:
            return 0.0
        j = bisect_right(values, x)
        lo_v, hi_v = values[j - 1], values[j]
        cdf = mids[j - 1] + (mids[j] - mids[j - 1]) * (x - lo_v) / (hi_v - lo_v) if hi_v > lo_v else mids[j]
        return 1.0 - cdf

    def tail(x: float) -> float:
        j = bisect_right(values, x)
        waiting = p_wait * cum[j - 1] * math.exp(-(x - values[j - 1]) / wait_scale_s) if j else 0.0
        return survival(x) + waiting

    target = 1.0 - q
    lo, hi = 0.0, values[-1] + wait_scale_s + 1e-9
    while tail(hi) > target:
        hi *= 2
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) > target else (lo, mid)
        if hi - lo <= 1e-4 * hi:
            break
    return hi
