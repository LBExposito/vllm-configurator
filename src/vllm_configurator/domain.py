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
    # Requêtes en vol visées (running + waiting), toutes instances confondues. Ce n'est pas un nombre
    # d'utilisateurs : un utilisateur agentique n'a une requête en cours qu'une partie du temps.
    target_concurrency: int | None = Field(default=None, gt=0)
    # Sessions en pause dont le préfixe doit rester en cache (offload KV, §5).
    paused_sessions: int | None = Field(default=None, ge=0)


class Slo(_Frozen):
    ttft_p95_ms: float | None = Field(default=None, gt=0)
    itl_p95_ms: float | None = Field(default=None, gt=0)
    # Ce que mesure la cible de latence de génération : « itl » = intervalle entre deux sorties d'une requête
    # (vllm:inter_token_latency_seconds, un step, plusieurs tokens avec le spec decoding) ; « tpot » = temps par
    # token (vllm:request_time_per_output_token_seconds), divisé par ~a avec le spec decoding.
    itl_metric: Literal["itl", "tpot"] = "itl"


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


class ObservedPerf(_Frozen):
    """Latencies measured in production with the generated configuration (Prometheus)."""

    itl_p95_ms: float | None = Field(default=None, gt=0)
    ttft_p95_ms: float | None = Field(default=None, gt=0)


class ObservedEpp(_Frozen):
    """Values read in the EPP log (« Prefix DataProducer initialized config={...} ») and per-pod vLLM metrics."""

    block_size_tokens: int | None = Field(default=None, gt=0)
    lru_capacity_per_server: int | None = Field(default=None, gt=0)
    max_prefix_tokens_to_match: int | None = Field(default=None, gt=0)
    # vllm:prefix_cache_hits / vllm:prefix_cache_queries, pire pod.
    prefix_cache_hit_rate: float | None = Field(default=None, ge=0, le=1)


RoutingScorer = Literal["queue-scorer", "kv-cache-utilization-scorer", "prefix-cache-scorer", "no-hit-lru-scorer"]


class RoutingRequest(_Frozen):
    """Inputs of the llm-d EPP target (smart routing between the vLLM instances described above)."""

    # None : la dernière version du catalogue llm-d.
    llmd_version: str | None = None
    # approximate : l'EPP estime le cache de chaque pod à partir de ce qu'il a routé ; precise : KV events ZMQ.
    prefix_mode: Literal["approximate", "precise"] = "approximate"
    # Préfixe commun à toutes les sessions (system prompt + outils), en tokens : il est en cache partout.
    shared_prefix_tokens: int | None = Field(default=None, ge=0)
    # Poids forcés du profil de scheduling (remplacent le poids calculé).
    weights: dict[RoutingScorer, int] = {}
    # Paramètres forcés du plugin de préfixe (approx-prefix-cache-producer, ou precise-prefix-cache-scorer en mode
    # precise) : « lruCapacityPerServer: 700000 ». Ils remplacent la valeur calculée ; un nom inconnu est ignoré.
    parameters: dict[str, Any] = {}
    # Mode precise : endpoint ZMQ auquel les pods vLLM publient leurs KV events.
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
    # Longueur de contexte servie (--max-model-len). None : contexte natif du modèle. Au-delà du natif, le moteur
    # applique l'extension de contexte du catalogue (YaRN via --hf-overrides) si le modèle en a une.
    max_model_len: int | None = Field(default=None, gt=0)
    # Flags vLLM supplémentaires (« --flag »: valeur ; dict → JSON, true → flag seul, false → --no-flag). Ils
    # remplacent un flag généré du même nom. Chaque flag est vérifié contre le code source de la version.
    extra_args: dict[str, Any] = {}
    # Variables d'environnement supplémentaires du conteneur vLLM.
    extra_env: dict[str, str] = {}
    # Réserve par GPU (activations, CUDA graphs, drafter) pour l'estimation du KV sans logs.
    kv_estimate_reserve_gib_per_gpu: float | None = Field(default=None, ge=0)
    # Routage llm-d (EPP) entre les instances : None = réglages par défaut de la cible llmd_epp.
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
    # Renseignés pour un flag hors du catalogue de la version (arguments supplémentaires).
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
    # Plage de l'estimation (incertitude du modèle de latence), quand elle s'applique.
    estimate_low: float | None = None
    estimate_high: float | None = None
    measured: float | None = None
    unit: str | None = None
    detail: str
    # False : affiché, mais n'entre pas dans le verdict global (cas pire, information).
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
    # Flag CLI → nouvelle valeur, quand la reco est un réglage.
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
    # Plages attendues (bas, haut) ; None si le modèle de latence est indisponible.
    itl_ms: tuple[float, float] | None = None
    tpot_ms: tuple[float, float] | None = None
    ttft_ms: tuple[float, float] | None = None
    output_tokens_per_s: tuple[float, float] | None = None
    kv_usage_pct: float | None = None
    prefill_load_pct: float | None = None
    expectation: CheckStatus = "unknown"
    expectation_reason: str = ""
    # Si ce palier est le dernier validé : instances (et GPU) pour la charge visée.
    instances_needed: int | None = None
    gpus_needed: int | None = None
    is_generated_config: bool = False


class TestPlan(_Frozen):
    steps: list[TestStep]
    summary: str
    protocol: list[str]
    pass_criteria: list[str]
    notes: list[str] = []


class SloAnalysis(_Frozen):
    verdict: CheckStatus
    summary: str
    checks: list[SloCheck]
    bottlenecks: list[Bottleneck]
    recommendations: list[Recommendation]
    # Décomposition du step au point de fonctionnement (ms), si le modèle de latence est disponible.
    step_breakdown: dict[str, Any] | None = None
    calibration: float | None = None
    # Multiplicateurs d'incertitude appliqués aux durées estimées (bas, haut).
    uncertainty: tuple[float, float] | None = None
    test_plan: TestPlan | None = None
    notes: list[str] = []
    confidence: Status = "unverified"


class GenerationResult(_Frozen):
    request: DeploymentRequest
    decisions: list[Decision]
    warnings: list[Warning]
    indicators: list[Indicator]
    assumptions: list[Assumption]
    facts: DeploymentFacts
    # Variables d'environnement à poser sur le conteneur vLLM (ex. VLLM_ALLOW_LONG_MAX_MODEL_LEN=1).
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
    # Plugins (type, parameters) et poids du profil, prêts à rendre en EndpointPickerConfig.
    plugins: list[dict[str, Any]] = []
    weights: dict[str, int] = {}
    picker: str | None = None
    # Commentaire court par paramètre, repris dans le YAML rendu.
    comments: dict[str, str] = {}
    # Flags à ajouter aux pods vLLM (mode precise : --kv-events-config).
    vllm_args: dict[str, Any] = {}
    # Lignes attendues dans les logs de l'EPP après déploiement.
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
