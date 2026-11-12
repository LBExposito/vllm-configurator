"""Scenario files: save and reload a deployment request (hardware, workload, SLOs, measurements) as YAML.

Format (version 1)::

    kind: vllm-configurator/scenario
    version: 1
    name: "Prod agentique B300"
    saved_at: "2026-11-12T10:00:00"
    request: { ... }        # DeploymentRequest, the only part read back on import
    generated: { ... }      # informational snapshot (args, verdict), ignored on import

A bare request (like tests/golden/reference_case/input.yaml) is also accepted on import.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import yaml
from pydantic import ValidationError

from .domain import DeploymentRequest, GenerationResult, RoutingRequest

KIND = "vllm-configurator/scenario"
VERSION = 1


class ScenarioError(ValueError):
    pass


def dump_scenario(request: DeploymentRequest, name: str | None = None, result: GenerationResult | None = None,
                  args_yaml: str | None = None, epp_yaml: str | None = None) -> str:
    doc: dict[str, Any] = {"kind": KIND, "version": VERSION}
    if name:
        doc["name"] = name
    doc["saved_at"] = datetime.now().isoformat(timespec="seconds")
    doc["request"] = request.model_dump(mode="json", exclude_none=True)
    if result is not None:
        generated: dict[str, Any] = {"note": "Instantané informatif, ignoré à l'import : la config est recalculée."}
        if result.env:
            generated["env"] = dict(result.env)
        if args_yaml:
            generated["args"] = yaml.safe_load(args_yaml)["args"]
        if epp_yaml:
            generated["llmd_epp"] = yaml.safe_load(epp_yaml)
        if result.analysis is not None:
            generated["slo_verdict"] = result.analysis.verdict
            generated["slo_summary"] = result.analysis.summary
            if result.analysis.test_plan is not None:
                generated["test_plan"] = result.analysis.test_plan.summary
        doc["generated"] = generated
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120)


def load_scenario(text: str | bytes) -> tuple[DeploymentRequest, str | None]:
    """Parse a scenario file (or a bare request). Returns the request and the scenario name."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"YAML invalide : {exc}") from exc
    if not isinstance(doc, dict):
        raise ScenarioError("Le fichier doit contenir un objet YAML (scénario ou requête).")
    name = None
    if "request" in doc or doc.get("kind") == KIND:
        if doc.get("kind") not in (None, KIND):
            raise ScenarioError(f"kind inattendu : {doc.get('kind')!r} (attendu {KIND!r}).")
        if doc.get("version", VERSION) > VERSION:
            raise ScenarioError(f"Version de scénario {doc['version']} plus récente que celle de l'app ({VERSION}).")
        name, data = doc.get("name"), doc.get("request")
        if not isinstance(data, dict):
            raise ScenarioError("Bloc « request » manquant.")
    else:
        data = doc
    try:
        return DeploymentRequest(**data), name
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))} : {e['msg']}" for e in exc.errors())
        raise ScenarioError(f"Scénario invalide : {details}") from exc


def request_to_form(request: DeploymentRequest) -> dict[str, Any]:
    """Session-state values of the Streamlit form for a request (inverse of the form → request mapping)."""
    wl, spec, kv, perf = request.workload, request.spec, request.observed_kv, request.observed_perf
    return {
        "gpu_id": request.gpu_id, "gpus_per_instance": request.gpus_per_instance, "num_instances": request.num_instances,
        "model_id": request.model_id, "vllm_version": request.vllm_version,
        "usage_type": wl.usage_type, "mode": wl.mode,
        "isl_p50": wl.isl.p50, "isl_p95": wl.isl.p95, "isl_p99": wl.isl.p99,
        "osl_p50": wl.osl.p50, "osl_p95": wl.osl.p95, "osl_p99": wl.osl.p99,
        "hit_rate": wl.prefix_cache_hit_rate, "target_concurrency": wl.target_concurrency,
        "paused_sessions": wl.paused_sessions,
        "ttft": request.slo.ttft_p95_ms, "itl": request.slo.itl_p95_ms, "itl_metric": request.slo.itl_metric,
        "itl_measured": perf.itl_p95_ms if perf else None, "ttft_measured": perf.ttft_p95_ms if perf else None,
        "allow_lossy": request.allow_lossy, "spec_enabled": spec.enabled, "k": spec.num_speculative_tokens,
        "draft_sample_method": spec.draft_sample_method or "défaut vLLM", "acceptance_length": spec.acceptance_length,
        "kv_tokens": kv.kv_cache_tokens if kv else None, "kv_util": kv.gpu_memory_utilization if kv else None,
        "kv_block": kv.block_size if kv else None,
        "gpu_util": request.gpu_memory_utilization, "reserve": request.kv_estimate_reserve_gib_per_gpu,
        "max_model_len": request.max_model_len,
        "extra_args_text": _yaml_text(request.extra_args), "extra_env_text": _yaml_text(request.extra_env),
        **routing_to_form(request.routing),
    }


ROUTING_WEIGHT_KEYS = {"queue-scorer": "llmd_w_queue", "kv-cache-utilization-scorer": "llmd_w_kv",
                       "prefix-cache-scorer": "llmd_w_prefix", "no-hit-lru-scorer": "llmd_w_nohit"}


def routing_to_form(routing: RoutingRequest | None) -> dict[str, Any]:
    """Session-state values of the « Routage llm-d » form section."""
    routing = routing or RoutingRequest()
    obs = routing.observed
    form: dict[str, Any] = {
        "llmd_version": routing.llmd_version, "llmd_mode": routing.prefix_mode,
        "llmd_shared_prefix": routing.shared_prefix_tokens, "llmd_kv_endpoint": routing.kv_events_endpoint or "",
        "llmd_obs_block": obs.block_size_tokens if obs else None,
        "llmd_obs_lru": obs.lru_capacity_per_server if obs else None,
        "llmd_obs_window": obs.max_prefix_tokens_to_match if obs else None,
        "llmd_obs_hit": obs.prefix_cache_hit_rate if obs else None,
        "llmd_params_text": _yaml_text(routing.parameters),
    }
    form.update({key: routing.weights.get(scorer) for scorer, key in ROUTING_WEIGHT_KEYS.items()})
    return form


def _yaml_text(mapping: dict[str, Any]) -> str:
    if not mapping:
        return ""
    return yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False, default_flow_style=None, width=1000)
