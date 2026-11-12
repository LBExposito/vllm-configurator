"""Render a GenerationResult as `vllm serve` arguments (KServe/K8s YAML and CLI)."""

from __future__ import annotations

import json
from typing import Any

from .catalog import Catalog, default_catalog
from .catalog.schema import FlagSpec
from .domain import Decision, GenerationResult

SECTIONS = [
    ("Parallélisme", ["tensor_parallel_size", "data_parallel_size"]),
    ("Scheduler", ["max_num_seqs", "max_num_batched_tokens", "long_prefill_token_threshold",
                   "long_prefill_token_threshold_adaptive", "enable_chunked_prefill"]),
    ("KV cache / prefix caching", ["enable_prefix_caching", "kv_cache_dtype", "gpu_memory_utilization",
                                   "block_size", "kv_offloading_size", "max_model_len"]),
    ("Speculative decoding", ["speculative_config"]),
    ("CUDA graphs", ["compilation_config"]),
]


def _flag_args(spec: FlagSpec, value: Any) -> list[str]:
    if spec.kind == "bool_optional":
        return [spec.cli if value else "--no-" + spec.cli[2:]]
    if spec.kind == "store_true":
        return [spec.cli] if value else []
    if spec.kind == "json":
        return [spec.cli, json.dumps(value, separators=(",", ":"), ensure_ascii=False)]
    return [spec.cli, str(value)]


def args_pairs(result: GenerationResult, catalog: Catalog | None = None) -> list[tuple[Decision, list[str]]]:
    """(decision, argv fragment) for every emitted decision, in section order."""
    catalog = catalog or default_catalog()
    flags = catalog.vllm_versions[result.request.vllm_version].flags
    cli_flags = set(catalog.extracted[result.request.vllm_version].cli_flags)
    by_key = {d.key: d for d in result.emitted}
    pairs = []
    for _, keys in SECTIONS:
        for key in keys:
            decision = by_key.get(key)
            if decision is None:
                continue
            spec = flags[key]
            if spec.cli not in cli_flags:
                raise ValueError(f"{spec.cli} absent de vLLM {result.request.vllm_version} : refus d'émettre un flag inconnu")
            pairs.append((decision, _flag_args(spec, decision.value)))
    return pairs


def argv(result: GenerationResult, catalog: Catalog | None = None) -> list[str]:
    return [token for _, fragment in args_pairs(result, catalog) for token in fragment]


def _yaml_scalar(token: str) -> str:
    if token.startswith("--"):
        return token
    if token.startswith("{"):
        return "'" + token.replace("'", "''") + "'"
    return json.dumps(token, ensure_ascii=False)


def render_k8s_args(result: GenerationResult, catalog: Catalog | None = None) -> str:
    """`args:` block for a KServe LLMInferenceService / Kubernetes container, with section comments."""
    pairs = {d.key: fragment for d, fragment in args_pairs(result, catalog)}
    omitted = {d.key: d for d in result.decisions if not d.emitted}
    lines = ["args:"]
    for title, keys in SECTIONS:
        body = []
        for key in keys:
            if key in pairs:
                body += [f"  - {_yaml_scalar(token)}" for token in pairs[key]]
            elif key in omitted and omitted[key].omitted_reason:
                body.append(f"  # pas de {key.replace('_', '-')} : {omitted[key].omitted_reason}")
        if body:
            lines.append(f"  # --- {title} ---")
            lines += body
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_cli(result: GenerationResult, catalog: Catalog | None = None) -> str:
    catalog = catalog or default_catalog()
    model = catalog.models[result.request.model_id]
    lines = [f"vllm serve {model.hf_repo}"]
    for _, fragment in args_pairs(result, catalog):
        rendered = [f"'{t}'" if t.startswith("{") else t for t in fragment]
        lines.append("  " + " ".join(rendered))
    return " \\\n".join(lines) + "\n"


def render_verify_command(result: GenerationResult, catalog: Catalog | None = None) -> str:
    """Shell command to check, on the deployed image, that every emitted flag exists."""
    flags = sorted({fragment[0] for _, fragment in args_pairs(result, catalog)})
    pattern = "|".join(flag.removeprefix("--no-").removeprefix("--") for flag in flags)
    return f"vllm serve --help=all | grep -oE -- \"--(no-)?({pattern})\\b\" | sort -u\n"
