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
    # None: the engine takes the value published for the usage type (displayed assumption) if there is one.
    prefix_cache_hit_rate: float | None = Field(default=None, ge=0, le=1)
    # Share of the requests whose prefix is missing from the cache (new session past the shared prefix, eviction,
    # misrouting): their prompt is prefilled in full. None: cold_request_fraction heuristic.
    cold_request_fraction: float | None = Field(default=None, ge=0, le=1)
    # Target requests in flight (running + waiting), all instances together. It is not a number of users: an agentic
    # user only has a request running part of the time.
    target_concurrency: int | None = Field(default=None, gt=0)
    # Paused sessions whose prefix must stay cached (KV offload, §5).
    paused_sessions: int | None = Field(default=None, ge=0)
    # Tokens to prefill for a warm request (what the turn adds: tool outputs, message). Sets the dispersion of their
    # distribution (read off a trace); None: uncached_tokens_sigma heuristic.
    warm_uncached_tokens: Percentiles | None = None
    # Mean wait between two requests of the same session (tools, a person), in seconds. Turns requests in flight into
    # sessions (test plan, working-set replay); None: test plan in requests in flight only.
    session_gap_s: float | None = Field(default=None, ge=0)


class Slo(_Frozen):
    ttft_p95_ms: float | None = Field(default=None, gt=0)
    itl_p95_ms: float | None = Field(default=None, gt=0)
    # What the generation latency target measures: « itl » = interval between two outputs of a request
    # (vllm:inter_token_latency_seconds, one step, several tokens with spec decoding); « tpot » = time per token
    # (vllm:request_time_per_output_token_seconds), divided by ~a with spec decoding.
    itl_metric: Literal["itl", "tpot"] = "itl"


class SpecOptions(_Frozen):
    enabled: bool = True
    num_speculative_tokens: int | None = Field(default=None, gt=0)
    draft_sample_method: Literal["greedy", "probabilistic"] | None = None
    # Measured acceptance length (Prometheus); otherwise the catalog assumption.
    acceptance_length: float | None = Field(default=None, ge=1)


class ObservedKv(_Frozen):
    """Values read in the vLLM startup log of one instance."""

    kv_cache_tokens: int = Field(gt=0, description="« GPU KV cache size: N tokens »")
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    block_size: int | None = Field(default=None, gt=0, description="« Setting kv cache block size to N »")


class ObservedPerf(_Frozen):
    """Latencies measured with the generated configuration (Prometheus, or a working-set replay step)."""

    itl_p95_ms: float | None = Field(default=None, gt=0)
    ttft_p95_ms: float | None = Field(default=None, gt=0)
    # Measured TPOT p95 when the target is the ITL: working-set measures the time per token of each request, not the
    # interval between two outputs. Recalibrates the decode part when no ITL is measured.
    tpot_p95_ms: float | None = Field(default=None, gt=0)
    # Requests in flight per instance during the measurement (running + waiting, mean). None: the target concurrency.
    inflight_per_instance: float | None = Field(default=None, gt=0)
    # working-set replay: sessions per instance of the measured step, and of the last passing step (measured capacity).
    sessions_per_instance: float | None = Field(default=None, gt=0)
    capacity_sessions_per_instance: float | None = Field(default=None, ge=0)


class ObservedEpp(_Frozen):
    """Values read in the EPP log (« Prefix DataProducer initialized config={...} ») and per-pod vLLM metrics."""

    block_size_tokens: int | None = Field(default=None, gt=0)
    lru_capacity_per_server: int | None = Field(default=None, gt=0)
    max_prefix_tokens_to_match: int | None = Field(default=None, gt=0)
    # vllm:prefix_cache_hits / vllm:prefix_cache_queries, worst pod.
    prefix_cache_hit_rate: float | None = Field(default=None, ge=0, le=1)
    # Share of the requests whose prefix is missing from the cache (new session past the shared prefix, eviction,
    # misrouting): their prompt is prefilled in full. None: cold_request_fraction heuristic.
    cold_request_fraction: float | None = Field(default=None, ge=0, le=1)


RoutingScorer = Literal["queue-scorer", "kv-cache-utilization-scorer", "prefix-cache-scorer", "no-hit-lru-scorer"]


class RoutingRequest(_Frozen):
    """Inputs of the llm-d EPP target (smart routing between the vLLM instances described above)."""

    # None: the latest llm-d version of the catalog.
    llmd_version: str | None = None
    # approximate: the EPP estimates each pod's cache from what it routed; precise: ZMQ KV events.
    prefix_mode: Literal["approximate", "precise"] = "approximate"
    # Prefix shared by all sessions (system prompt + tools), in tokens: it is cached everywhere.
    shared_prefix_tokens: int | None = Field(default=None, ge=0)
    # Forced weights of the scheduling profile (they replace the computed weight).
    weights: dict[RoutingScorer, int] = {}
    # Forced parameters of the prefix plugin (approx-prefix-cache-producer, or precise-prefix-cache-scorer in precise
    # mode): « lruCapacityPerServer: 700000 ». They replace the computed value; an unknown name is ignored.
    parameters: dict[str, Any] = {}
    # Precise mode: ZMQ endpoint the vLLM pods publish their KV events to.
    kv_events_endpoint: str | None = None
    observed: ObservedEpp | None = None


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
    observed_perf: ObservedPerf | None = None
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    # Served context length (--max-model-len). None: native context of the model. Beyond the native one, the engine
    # applies the catalog's context extension (YaRN via --hf-overrides) if the model has one.
    max_model_len: int | None = Field(default=None, gt=0)
    # Extra vLLM flags (« --flag »: value; dict → JSON, true → bare flag, false → --no-flag). They replace a generated
    # flag of the same name. Each flag is checked against the source code of the version.
    extra_args: dict[str, Any] = {}
    # Extra environment variables of the vLLM container.
    extra_env: dict[str, str] = {}
    # Reserve per GPU (activations, CUDA graphs, drafter) for the KV estimate without logs.
    kv_estimate_reserve_gib_per_gpu: float | None = Field(default=None, ge=0)
    # CPU cores available per instance (pod CPU limit): caps the API processes and the renderer threads.
    cpu_cores_per_instance: float | None = Field(default=None, gt=0)
    # llm-d (EPP) routing between the instances: None = default settings of the llmd_epp target.
    routing: RoutingRequest | None = None


# --- Outputs ---------------------------------------------------------------

Severity = Literal["info", "warning", "critical"]
Category = Literal[
    "to_verify", "lossy", "osl_suspect", "isl_tail", "kv_limited", "unsupported", "breaking_change",
    "assumption", "itl", "memory", "version", "model_note", "context", "extra_args", "routing",
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
    # Filled in for a flag outside the version's catalog (extra arguments).
    cli: str | None = None
    cli_kind: Literal["value", "bool_optional", "store_true", "json"] | None = None


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


CheckStatus = Literal["ok", "at_risk", "ko", "unknown"]


class SloCheck(_Frozen):
    key: str
    label: str
    status: CheckStatus
    target: float | None = None
    estimate: float | None = None
    # Range of the estimate (latency model uncertainty), when it applies.
    estimate_low: float | None = None
    estimate_high: float | None = None
    measured: float | None = None
    unit: str | None = None
    detail: str
    # False: displayed, but not part of the overall verdict (worst case, information).
    in_verdict: bool = True


class Bottleneck(_Frozen):
    key: str
    label: str
    status: CheckStatus
    explanation: str


class Recommendation(_Frozen):
    kind: Literal["none", "param", "capacity", "architecture", "measure"]
    title: str
    rationale: str
    # CLI flag → new value, when the recommendation is a setting.
    changes: dict[str, Any] = {}
    addresses: list[str] = []


class TestStep(_Frozen):
    """One load step of the validation plan: deploy this setting, inject this load, compare to the SLOs."""

    index: int
    max_num_seqs: int
    max_num_batched_tokens: int
    long_prefill_token_threshold: int
    capture_size: int
    inflight_per_instance: int
    # Expected ranges (low, high); None if the latency model is unavailable.
    itl_ms: tuple[float, float] | None = None
    tpot_ms: tuple[float, float] | None = None
    ttft_ms: tuple[float, float] | None = None
    output_tokens_per_s: tuple[float, float] | None = None
    kv_usage_pct: float | None = None
    prefill_load_pct: float | None = None
    expectation: CheckStatus = "unknown"
    expectation_reason: str = ""
    # If this step is the last one to pass: instances (and GPUs) for the target load.
    instances_needed: int | None = None
    gpus_needed: int | None = None
    is_generated_config: bool = False
    # Equivalent sessions per instance (working-set replay): requests in flight × (duration + wait) / duration, for the
    # longest then the shortest request duration of the estimate. None without a wait between requests.
    sessions_per_instance: tuple[int, int] | None = None
    # KV held by these sessions (active and waiting), at the low end, in % of one instance's pool.
    sessions_kv_pct: float | None = None


class TestPlan(_Frozen):
    steps: list[TestStep]
    summary: str
    protocol: list[str]
    pass_criteria: list[str]
    notes: list[str] = []


class SessionCapacity(_Frozen):
    """Agentic sessions per instance that meet the SLOs: the configuration as generated, and the best vLLM setting found."""

    current_sessions: int
    current_inflight: int
    current_binding: str
    best_sessions: int
    best_inflight: int
    best_binding: str
    # Flags to change to go from the config to the best setting (empty: the config already is the best setting).
    changes: dict[str, Any] = {}
    # Capacity validated by working-set, per instance (measurement), for comparison.
    measured_sessions: float | None = None
    summary: str
    notes: list[str] = []


class SloAnalysis(_Frozen):
    verdict: CheckStatus
    summary: str
    checks: list[SloCheck]
    bottlenecks: list[Bottleneck]
    recommendations: list[Recommendation]
    # Breakdown of the step at the operating point (ms), if the latency model is available.
    step_breakdown: dict[str, Any] | None = None
    # Calibration factors: decode part (measured generation latency), prefill part (measured TTFT).
    calibration: float | None = None
    calibration_prefill: float | None = None
    # Uncertainty multipliers (low, high) of the generation latency and of the TTFT.
    uncertainty: tuple[float, float] | None = None
    uncertainty_ttft: tuple[float, float] | None = None
    test_plan: TestPlan | None = None
    # Capacity in agentic sessions (if the wait between two requests of a session is known).
    session_capacity: SessionCapacity | None = None
    notes: list[str] = []
    confidence: Status = "unverified"


class GenerationResult(_Frozen):
    request: DeploymentRequest
    decisions: list[Decision]
    warnings: list[Warning]
    indicators: list[Indicator]
    assumptions: list[Assumption]
    facts: DeploymentFacts
    # Environment variables to set on the vLLM container (e.g. VLLM_ALLOW_LONG_MAX_MODEL_LEN=1).
    env: dict[str, str] = {}
    analysis: SloAnalysis | None = None
    blocking_errors: list[str] = []

    def decision(self, key: str) -> Decision:
        for d in self.decisions:
            if d.key == key:
                return d
        raise KeyError(key)

    @property
    def emitted(self) -> list[Decision]:
        return [d for d in self.decisions if d.emitted]


class RoutingResult(_Frozen):
    """llm-d EPP configuration derived from the vLLM GenerationResult (DeploymentFacts)."""

    llmd_version: str
    prefix_mode: Literal["approximate", "precise"]
    decisions: list[Decision]
    warnings: list[Warning]
    indicators: list[Indicator]
    assumptions: list[Assumption]
    # Plugins (type, parameters) and profile weights, ready to render as an EndpointPickerConfig.
    plugins: list[dict[str, Any]] = []
    weights: dict[str, int] = {}
    picker: str | None = None
    # Short comment per parameter, carried into the rendered YAML.
    comments: dict[str, str] = {}
    # Flags to add to the vLLM pods (precise mode: --kv-events-config).
    vllm_args: dict[str, Any] = {}
    # Lines expected in the EPP logs after deployment.
    expected_logs: list[str] = []
    blocking_errors: list[str] = []

    def decision(self, key: str) -> Decision:
        for d in self.decisions:
            if d.key == key:
                return d
        raise KeyError(key)


class GenerationError(Exception):
    """Raised when a required input is missing or the catalog does not cover the request."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors
