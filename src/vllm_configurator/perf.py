"""First-order latency model of one vLLM scheduler step (roofline), used by the SLO analysis.

A step processes T = decode tokens + prefill tokens. Its duration is bounded by
- memory: weights read once per step (for an MoE, only the experts hit by at least one token)
  plus the KV of every decoding sequence, at the effective HBM bandwidth of one GPU;
- compute: 2 × activated parameters per token, at the effective tensor-core throughput;
plus a fixed overhead per step (kernel launches, TP all-reduce, scheduler) proportional to the
number of layers. All quantities are per GPU: weights, FLOPs and KV are split over the TP ranks
(the KV per GPU is passed in as is, already taking replication into account).

This is an order of magnitude, not a prediction: the analysis labels every value derived from
it as « à vérifier » and lets a measured ITL recalibrate it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PerfModel:
    tp: int
    hbm_bytes_per_s: float          # effective, per GPU
    flops_per_s: float              # effective, per GPU
    overhead_s: float               # fixed per step
    active_params_decode: float     # per output token
    active_params_prefill: float    # per prompt token
    dense_params: float             # non-routed weights read at every step
    dense_bytes_per_param: float
    routed_params: float            # all routed experts (0 for a dense model)
    routed_bytes_per_param: float
    experts_per_tok: int
    n_experts: int
    kv_bytes_per_token_per_gpu: float | None
    calibration: float = 1.0

    def calibrated(self, factor: float) -> "PerfModel":
        return replace(self, calibration=factor)


@dataclass(frozen=True)
class StepTime:
    seconds: float
    memory_s: float
    compute_s: float
    overhead_s: float
    weights_bytes: float
    kv_bytes: float

    @property
    def bound(self) -> str:
        """« memory » or « compute »: the roofline term that dominates the step."""
        return "memory" if self.memory_s >= self.compute_s else "compute"

    @property
    def ms(self) -> float:
        return self.seconds * 1e3


def experts_hit_fraction(tokens: float, experts_per_tok: int, n_experts: int) -> float:
    """Expected fraction of routed experts hit by at least one of `tokens` tokens (uniform routing)."""
    if n_experts <= 0 or tokens <= 0:
        return 0.0
    return 1.0 - (1.0 - experts_per_tok / n_experts) ** tokens


def step_time(m: PerfModel, decode_seqs: int, tokens_per_seq: int, context_tokens: float, prefill_tokens: float) -> StepTime:
    """Duration of one step with `decode_seqs` sequences in decode (each verifying `tokens_per_seq`
    tokens and reading `context_tokens` of KV) and `prefill_tokens` prompt tokens."""
    decode_tokens = decode_seqs * tokens_per_seq
    tokens = decode_tokens + prefill_tokens
    hit = experts_hit_fraction(tokens, m.experts_per_tok, m.n_experts) if m.routed_params else 0.0
    weights = (m.dense_params * m.dense_bytes_per_param + m.routed_params * hit * m.routed_bytes_per_param) / m.tp
    kv = decode_seqs * context_tokens * (m.kv_bytes_per_token_per_gpu or 0.0)
    memory = (weights + kv) / m.hbm_bytes_per_s
    flops = 2.0 * (m.active_params_decode * decode_tokens + m.active_params_prefill * prefill_tokens) / m.tp
    compute = flops / m.flops_per_s
    total = (max(memory, compute) + m.overhead_s) * m.calibration
    return StepTime(seconds=total, memory_s=memory * m.calibration, compute_s=compute * m.calibration,
                    overhead_s=m.overhead_s * m.calibration, weights_bytes=weights, kv_bytes=kv)


def md1_wait(rho: float, service_s: float) -> float:
    """Mean wait of an M/D/1 queue (Pollaczek–Khinchine): rho / (2 (1 − rho)) × service. inf if rho >= 1."""
    if rho >= 1.0:
        return math.inf
    return rho / (2.0 * (1.0 - rho)) * service_s
