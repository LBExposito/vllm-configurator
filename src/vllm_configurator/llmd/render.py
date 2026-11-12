"""Render a RoutingResult as an EndpointPickerConfig (YAML with comments) and the vLLM args it needs."""

from __future__ import annotations

import json
from typing import Any

from ..catalog import Catalog, default_catalog
from ..domain import RoutingResult


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _with_comment(line: str, comment: str | None, column: int = 44) -> str:
    return f"{line.ljust(column)} # {comment}" if comment else line


def _params(params: dict[str, Any], comments: dict[str, str], indent: int) -> list[str]:
    lines = []
    pad = " " * indent
    for key, value in params.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines += _params(value, {k.removeprefix(f"{key}."): v for k, v in comments.items()}, indent + 2)
        else:
            lines.append(_with_comment(f"{pad}{key}: {_scalar(value)}", comments.get(key)))
    return lines


def render_epp_config(routing: RoutingResult, catalog: Catalog | None = None) -> str:
    """EndpointPickerConfig for the LLMInferenceService scheduler (empty string if generation is blocked)."""
    if routing.blocking_errors:
        return ""
    catalog = catalog or default_catalog()
    llmd = catalog.llmd_versions[routing.llmd_version]
    comments = {**routing.comments, **{f"{key.split('.', 1)[-1]}": c for key, c in routing.comments.items() if "." in key}}
    lines = [f"apiVersion: {llmd.api_version}", f"kind: {llmd.kind}", "plugins:"]
    for plugin in routing.plugins:
        note = "🔶 porteur des paramètres selon la version" if plugin["type"] == "approx-prefix-cache-producer" else None
        lines.append(_with_comment(f"  - type: {plugin['type']}", note))
        if plugin.get("parameters"):
            lines.append("    parameters:")
            lines += _params(plugin["parameters"], comments, 6)
    lines += ["schedulingProfiles:", "  - name: default", "    plugins:"]
    defaults = llmd.default_profile.weights
    for scorer, weight in routing.weights.items():
        default = defaults.get(scorer if scorer in defaults else "prefix-cache-scorer")
        lines.append(f"      - pluginRef: {scorer}")
        lines.append(_with_comment(f"        weight: {weight}", f"défaut RHOAI {default}" if default != weight else None))
    lines.append(f"      - pluginRef: {routing.picker}")
    return "\n".join(lines) + "\n"


def render_vllm_args(routing: RoutingResult) -> str:
    """Args to append to the vLLM container (precise mode); empty string when vLLM needs nothing more."""
    if not routing.vllm_args:
        return ""
    lines = ["args:  # à ajouter aux args vLLM"]
    for flag, value in routing.vllm_args.items():
        rendered = json.dumps(value, separators=(",", ":"), ensure_ascii=False) if isinstance(value, dict) else str(value)
        lines += [f"  - {flag}", "  - '" + rendered.replace("'", "''") + "'"]
    return "\n".join(lines) + "\n"
