"""Shared inputs of the rules and the container they return."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..catalog.schema import Behavior, ExtractedVersion, Gpu, Heuristic, Model, VllmVersion, WorkloadCatalog, WorkloadProfile
from ..domain import Assumption, Decision, DeploymentRequest, Indicator, Warning

KB = "docs/vllm-config-knowledge-base.md"


@dataclass(frozen=True)
class RuleContext:
    request: DeploymentRequest
    gpu: Gpu
    model: Model
    vllm: VllmVersion
    extracted: ExtractedVersion
    workload: WorkloadCatalog

    @property
    def profile(self) -> WorkloadProfile:
        return self.workload.profiles[self.request.workload.usage_type]

    def behavior(self, key: str) -> Behavior:
        try:
            return self.vllm.behaviors[key]
        except KeyError as exc:
            raise KeyError(f"comportement {key!r} absent du catalogue vLLM {self.vllm.version}") from exc

    def default(self, key: str) -> Any:
        entry = self.vllm.defaults.get(key)
        return entry.value if entry else None

    def heuristic(self, key: str) -> Heuristic:
        return self.workload.heuristics[key]

    def has_flag(self, key: str) -> bool:
        spec = self.vllm.flags.get(key)
        return spec is not None and spec.cli in self.extracted.cli_flags


@dataclass
class RuleOutput:
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)
    indicators: list[Indicator] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    # Values made available to the following rules (key -> value).
    state: dict[str, Any] = field(default_factory=dict)

    def extend(self, other: "RuleOutput") -> None:
        self.decisions += other.decisions
        self.warnings += other.warnings
        self.indicators += other.indicators
        self.assumptions += other.assumptions
        self.state.update(other.state)


def fmt_int(value: float) -> str:
    """French thousands separator (narrow no-break space)."""
    return f"{round(value):,}".replace(",", " ")


def unverified_warning(what: str, source: str | list[str], rule_id: str, note: str | None = None) -> Warning:
    src = source if isinstance(source, str) else " ; ".join(source)
    message = f"À vérifier : {what}." + (f" {note}" if note else "")
    return Warning(severity="warning", category="to_verify", message=message, source=src, rule_id=rule_id)
