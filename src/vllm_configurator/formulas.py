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
