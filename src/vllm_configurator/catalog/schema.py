"""Pydantic schemas of the versioned catalogs (catalog/**/*.yaml).

Every block that carries facts has a `status` and a `source`. `verified` means
read in a primary source (vLLM source code at a tag, Hugging Face config.json
at a pinned revision, vendor datasheet, real startup logs). Anything else is
`unverified` and is surfaced by the engine as a "à vérifier" warning.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Status = Literal["verified", "unverified"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(_Strict):
    status: Status
    source: str | list[str]
    note: str | None = None


# --- GPU -------------------------------------------------------------------

class GpuMemory(Provenance):
    hbm_gb: float = Field(gt=0, description="Capacité HBM par GPU, en Go (10^9 octets)")
    hbm_type: str | None = None


class GpuBandwidth(Provenance):
    hbm_tb_s: float | None = Field(default=None, gt=0)
    nvlink_gpu_to_gpu_tb_s: float | None = Field(default=None, gt=0)


class GpuComputeCapability(Provenance):
    value: str


ComputeDtype = Literal["fp4", "fp8", "bf16"]


class GpuCompute(Provenance):
    """Dense tensor-core peak per GPU, in TFLOPS (no sparsity)."""

    dense_tflops: dict[ComputeDtype, float]


class Gpu(_Strict):
    id: str
    display_name: str
    vendor: Literal["nvidia", "amd", "other"]
    generation: str
    # Clé de cudagraph.default_max_capture_size dans le catalogue vLLM.
    cudagraph_family: Literal["blackwell_dc", "other"]
    compute_capability: GpuComputeCapability
    memory: GpuMemory
    bandwidth: GpuBandwidth | None = None
    # Utilisé seulement par le modèle de latence de l'analyse SLO (roofline).
    compute: GpuCompute | None = None
    verified_on: str
    notes: list[str] = []


# --- Model -----------------------------------------------------------------

class MoeSpec(_Strict):
    n_routed_experts: int = Field(gt=0)
    num_experts_per_tok: int = Field(gt=0)
    n_shared_experts: int = Field(default=0, ge=0)


class Architecture(Provenance):
    hf_architecture: str
    model_type: str
    family: Literal["dense", "moe"]
    attention: list[str] = Field(min_length=1, description="Types d'attention, ex. gqa, mla, sparse, sliding_window, linear")
    num_hidden_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int | None = Field(default=None, gt=0)
    head_dim: int | None = Field(default=None, ge=0)
    max_position_embeddings: int = Field(gt=0)
    moe: MoeSpec | None = None
    # Champs bruts de config.json utiles mais non interprétés par le moteur.
    raw: dict[str, Any] = {}

    @model_validator(mode="after")
    def _moe_consistency(self) -> "Architecture":
        if (self.family == "moe") != (self.moe is not None):
            raise ValueError("family='moe' exige un bloc moe, et réciproquement")
        return self


class Weights(Provenance):
    checkpoint_bytes: int | None = Field(default=None, gt=0)
    quantization: str | None = None


class KvCache(Provenance):
    # null = non dérivable de façon fiable : l'app exige alors la taille lue dans les logs.
    bytes_per_token: float | None = Field(default=None, gt=0)
    bytes_per_token_dtype: str | None = None
    # Sommes « par token » répliquées sur chaque rang TP (MLA, MQA) ou partagées.
    replicated_across_tp: bool | None = None
    # État fixe par séquence (attention linéaire, fenêtre glissante), si connu.
    per_sequence_state_bytes: float | None = Field(default=None, ge=0)
    method: str = Field(description="Formule ou raison de l'absence de valeur")
    recommended_dtype: str | None = None
    # Provenance propre au dtype recommandé (souvent une recette vLLM), distincte de celle de bytes_per_token.
    recommended_dtype_status: Status = "unverified"
    recommended_dtype_source: str | None = None


class BlockSize(Provenance):
    # null = laissé au backend ; ne jamais forcer sans raison.
    forced_by_backend: int | None = Field(default=None, gt=0)


class Speculative(Provenance):
    native_method: str | None
    recommended_num_speculative_tokens: int | None = Field(default=None, gt=0)
    fixed_k: bool = False


class ModelPerformance(Provenance):
    """Inputs of the roofline latency model used by the SLO analysis (analysis.py)."""

    method: str = Field(description="Comment les valeurs ont été obtenues")

    # Paramètres activés par token (FLOPs ≈ 2 × paramètres activés), hors tables de lookup (engram, n-gram).
    active_params_per_output_token: float = Field(gt=0)
    active_params_per_prompt_token: float | None = Field(default=None, gt=0)
    # MoE : taille des experts routés, pour estimer les octets de poids lus par step selon le nombre de tokens.
    moe_layers: int | None = Field(default=None, gt=0)
    moe_intermediate_size: int | None = Field(default=None, gt=0)
    routed_expert_bytes_per_param: float | None = Field(default=None, gt=0)
    # Octets par paramètre des poids non routés (attention, expert partagé, couches denses, lm_head).
    dense_bytes_per_param: float = Field(gt=0)
    # Type des tensor cores utilisés par les GEMM (débit crête du GPU pris pour ce type).
    compute_dtype: ComputeDtype


class ContextExtension(Provenance):
    """RoPE scaling that lets the model serve contexts beyond its native length (--hf-overrides)."""

    method: Literal["yarn"]
    native_max_model_len: int = Field(gt=0)
    max_model_len: int = Field(gt=0)
    override_path: list[str] = Field(min_length=1)
    # Bloc complet, sans `factor` : le moteur calcule le facteur d'après la longueur visée.
    rope_parameters: dict[str, Any]


class Serving(Provenance):
    min_vllm_version: str | None = None
    requires_custom_image: str | None = None


class Model(_Strict):
    id: str
    display_name: str
    hf_repo: str
    hf_revision: str = Field(min_length=40, max_length=40)
    verified_on: str
    architecture: Architecture
    weights: Weights
    kv_cache: KvCache
    block_size: BlockSize
    speculative: Speculative
    serving: Serving
    performance: ModelPerformance | None = None
    context_extension: ContextExtension | None = None
    notes: list[str] = []

    @model_validator(mode="after")
    def _performance_moe(self) -> "Model":
        perf = self.performance
        if perf is not None and self.architecture.moe is not None:
            if None in (perf.moe_layers, perf.moe_intermediate_size, perf.routed_expert_bytes_per_param):
                raise ValueError("performance d'un MoE : moe_layers, moe_intermediate_size et routed_expert_bytes_per_param requis")
        return self


# --- vLLM version ----------------------------------------------------------

class Behavior(Provenance):
    value: Any


class FlagSpec(_Strict):
    cli: str = Field(pattern=r"^--[a-z0-9][a-z0-9-]*$")
    kind: Literal["value", "bool_optional", "store_true", "json"]


class ReleaseItem(_Strict):
    id: str
    text: str
    status: Status


class VllmVersion(_Strict):
    version: str
    release_date: str | None = None
    source_tag: str
    verified_on: str
    defaults: dict[str, Behavior]
    behaviors: dict[str, Behavior]
    flags: dict[str, FlagSpec]
    breaking_changes: list[ReleaseItem] = []
    notes: list[ReleaseItem] = []


# --- Workload profiles, heuristics, metrics ---------------------------------

UsageType = Literal["chat", "agentic", "rag", "extraction", "long_summary", "scoring", "batch_offline"]


class SourcedValue(Provenance):
    value: float


class WorkloadProfile(_Strict):
    label: str
    isl_typical: str
    osl_typical: str
    ratio_range: tuple[float, float] | None = None
    fills_16k: str
    recommended_budget: int = Field(gt=0)
    budget_range: tuple[int, int]
    source: str
    reference_ratio: SourcedValue | None = None
    prefix_cache_hit_rate: SourcedValue | None = None
    osl_per_call_median: SourcedValue | None = None
    tool_calls: bool = False


class Heuristic(Provenance):
    value: Any


class Metric(_Strict):
    name: str
    usage: str
    note: str | None = None


class WorkloadCatalog(_Strict):
    profiles: dict[UsageType, WorkloadProfile]
    heuristics: dict[str, Heuristic]
    metrics: list[Metric]
    validation_tests: list[str] = []


class ExtractedVersion(BaseModel):
    """Generated by tools/extract_vllm_flags.py."""

    version: str
    source: dict[str, Any]
    cli_flags: list[str]
    speculative_config_keys: list[str]
    speculative_methods: list[str]
    rejection_sample_methods: list[str]
    draft_sample_methods: list[str]
    compilation_config_keys: list[str]
    architectures: list[str]
