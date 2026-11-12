"""Run the rules pipeline for a deployment request."""

from __future__ import annotations

from typing import Any

from .analysis import analyze
from .catalog import Catalog, default_catalog
from .domain import DeploymentFacts, DeploymentRequest, GenerationError, GenerationResult
from .rules import PIPELINE, RuleContext, RuleOutput

# Display order of the decisions (args order in the rendered config).
DECISION_ORDER = [
    "tensor_parallel_size", "data_parallel_size",
    "max_num_seqs", "max_num_batched_tokens", "long_prefill_token_threshold", "long_prefill_token_threshold_adaptive",
    "enable_chunked_prefill",
    "enable_prefix_caching", "kv_cache_dtype", "gpu_memory_utilization", "block_size", "kv_offloading_size",
    "max_model_len",
    "speculative_config", "compilation_config",
]


def build_context(request: DeploymentRequest, catalog: Catalog) -> RuleContext:
    errors = []
    if request.gpu_id not in catalog.gpus:
        errors.append(f"GPU {request.gpu_id!r} absent du catalogue (catalog/gpus/). Disponibles : {sorted(catalog.gpus)}")
    if request.model_id not in catalog.models:
        errors.append(f"Modèle {request.model_id!r} absent du catalogue (catalog/models/). Disponibles : {sorted(catalog.models)}")
    if request.vllm_version not in catalog.vllm_versions or request.vllm_version not in catalog.extracted:
        errors.append(
            f"vLLM {request.vllm_version} absent du catalogue (catalog/vllm/ et catalog/vllm/extracted/)."
            f" Disponibles : {catalog.sorted_versions()}. Voir le README pour ajouter une version."
        )
    if errors:
        raise GenerationError(errors)
    return RuleContext(
        request=request, gpu=catalog.gpus[request.gpu_id], model=catalog.models[request.model_id],
        vllm=catalog.vllm_versions[request.vllm_version], extracted=catalog.extracted[request.vllm_version],
        workload=catalog.workload,
    )


def _run_pipeline(ctx: RuleContext, util_hint: float) -> RuleOutput:
    total = RuleOutput(state={"gpu_memory_utilization_hint": util_hint})
    for rule in PIPELINE:
        if total.state.get("blocking"):
            break
        total.extend(rule(ctx, dict(total.state)))
    return total


def generate(request: DeploymentRequest, catalog: Catalog | None = None) -> GenerationResult:
    """Generate the vLLM configuration. Raises GenerationError if the catalog does not cover the request."""
    catalog = catalog or default_catalog()
    ctx = build_context(request, catalog)

    hint = request.gpu_memory_utilization or ctx.default("gpu_memory_utilization")
    out = _run_pipeline(ctx, hint)
    # Without startup logs, the KV estimate depends on gpu_memory_utilization, which itself depends on the
    # CUDA graph capture size: rerun once with the chosen value.
    chosen = out.state.get("gpu_memory_utilization")
    if out.state.get("kv_source") == "estimate" and chosen is not None and chosen != hint:
        out = _run_pipeline(ctx, chosen)

    order = {key: i for i, key in enumerate(DECISION_ORDER)}
    decisions = sorted(out.decisions, key=lambda d: order.get(d.key, len(order)))
    severity = {"critical": 0, "warning": 1, "info": 2}
    warnings = sorted(out.warnings, key=lambda w: severity[w.severity])
    s: dict[str, Any] = out.state
    facts = DeploymentFacts(
        model_id=request.model_id, vllm_version=request.vllm_version, num_instances=request.num_instances,
        tensor_parallel_size=request.gpus_per_instance,
        kv_tokens_per_instance=s.get("kv_tokens"), kv_tokens_source=s.get("kv_source", "unknown"),
        kv_tokens_measured_with_gpu_memory_utilization=(
            request.observed_kv.gpu_memory_utilization if request.observed_kv else s.get("gpu_memory_utilization")),
        block_size=s.get("block_size"), block_size_source=s.get("block_size_source", "unknown"),
        max_num_seqs=s.get("max_num_seqs"), max_num_batched_tokens=s.get("max_num_batched_tokens"),
        num_speculative_tokens=s.get("num_speculative_tokens", 0), gpu_memory_utilization=s.get("gpu_memory_utilization"),
    )
    blocking = list(s.get("blocking", []))
    return GenerationResult(
        request=request, decisions=decisions, warnings=warnings, indicators=out.indicators,
        assumptions=out.assumptions, facts=facts, analysis=None if blocking else analyze(ctx, s),
        blocking_errors=blocking,
    )
