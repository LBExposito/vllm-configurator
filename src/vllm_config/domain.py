"""Input and output models of the rules engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog.schema import Status, UsageType


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Inputs ----------------------------------------------------------------

class Percentiles(_Frozen):
    p50: int = Field(gt=0)
    p95: int | None = Field(default=None, gt=0)
    p99: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Percentiles":
        values = [v for v in (self.p50, self.p95, self.p99) if v is not None]
        if values != sorted(values):
            raise ValueError("les percentiles doivent être croissants (P50 ≤ P95 ≤ P99)")
        return self


class Workload(_Frozen):
    usage_type: UsageType
    mode: Literal["interactive", "batch"] = "interactive"
    isl: Percentiles
    osl: Percentiles
    # None : le moteur prend la valeur publiée pour le type d'usage (hypothèse affichée) s'il y en a une.
    prefix_cache_hit_rate: float | None = Field(default=None, ge=0, le=1)
    # Concurrence cible totale (toutes instances confondues).
    target_concurrency: int | None = Field(default=None, gt=0)
    # Sessions en pause dont le préfixe doit rester en cache (offload KV, §5).
    paused_sessions: int | None = Field(default=None, ge=0)


class Slo(_Frozen):
    ttft_p95_ms: float | None = Field(default=None, gt=0)
    itl_p95_ms: float | None = Field(default=None, gt=0)


class SpecOptions(_Frozen):
    enabled: bool = True
    num_speculative_tokens: int | None = Field(default=None, gt=0)
    draft_sample_method: Literal["greedy", "probabilistic"] | None = None
    # Longueur d'acceptation mesurée (Prometheus) ; sinon hypothèse du catalogue.
    acceptance_length: float | None = Field(default=None, ge=1)


class ObservedKv(_Frozen):
    """Values read in the vLLM startup log of one instance."""

    kv_cache_tokens: int = Field(gt=0, description="« GPU KV cache size: N tokens »")
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    block_size: int | None = Field(default=None, gt=0, description="« Setting kv cache block size to N »")


class DeploymentRequest(_Frozen):
    gpu_id: str
    gpus_per_instance: int = Field(gt=0)
    num_instances: int = Field(default=1, gt=0)
    model_id: str
    vllm_version: str
    workload: Workload
    slo: Slo = Slo()
    allow_lossy: bool = False
    spec: SpecOptions = SpecOptions()
    observed_kv: ObservedKv | None = None
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    # Réserve par GPU (activations, CUDA graphs, drafter) pour l'estimation du KV sans logs.
    kv_estimate_reserve_gib_per_gpu: float | None = Field(default=None, ge=0)


# --- Outputs ---------------------------------------------------------------

Severity = Literal["info", "warning", "critical"]
Category = Literal[
    "to_verify", "lossy", "osl_suspect", "isl_tail", "kv_limited", "unsupported", "breaking_change",
    "assumption", "itl", "memory", "version", "model_note",
]


class Decision(_Frozen):
    key: str
    value: Any
    rule_id: str
    title: str
    justification: str
    formula: str | None = None
    computation: str | None = None
    confidence: Status = "verified"
    sources: list[str] = []
    default: Any = None
    emitted: bool = True
    omitted_reason: str | None = None


class Warning(_Frozen):
    severity: Severity
    category: Category
    message: str
    source: str | None = None
    rule_id: str | None = None


class Indicator(_Frozen):
    key: str
    label: str
    value: float | int | str | None
    unit: str | None = None
    formula: str | None = None
    computation: str | None = None
    interpretation: str | None = None
    confidence: Status = "verified"


class Assumption(_Frozen):
    key: str
    label: str
    value: Any
    reason: str
    status: Status = "unverified"


class DeploymentFacts(_Frozen):
    """Stable interface consumed by other targets (v2: llm-d EPP)."""

    model_id: str
    vllm_version: str
    num_instances: int
    tensor_parallel_size: int
    kv_tokens_per_instance: int | None
    kv_tokens_source: Literal["startup_log", "estimate", "unknown"]
    kv_tokens_measured_with_gpu_memory_utilization: float | None
    block_size: int | None
    block_size_source: Literal["startup_log", "backend_forced", "unknown"]
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    num_speculative_tokens: int
    gpu_memory_utilization: float | None


class GenerationResult(_Frozen):
    request: DeploymentRequest
    decisions: list[Decision]
    warnings: list[Warning]
    indicators: list[Indicator]
    assumptions: list[Assumption]
    facts: DeploymentFacts
    blocking_errors: list[str] = []

    def decision(self, key: str) -> Decision:
        for d in self.decisions:
            if d.key == key:
                return d
        raise KeyError(key)

    @property
    def emitted(self) -> list[Decision]:
        return [d for d in self.decisions if d.emitted]


class GenerationError(Exception):
    """Raised when a required input is missing or the catalog does not cover the request."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors
