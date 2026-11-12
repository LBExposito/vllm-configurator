"""Additional vLLM arguments and environment variables supplied by the user."""

from __future__ import annotations

import json
from typing import Any

from ..domain import Decision, Warning
from .context import RuleContext, RuleOutput

# Flags dont l'app tire des conséquences : les passer en argument supplémentaire contourne ces calculs.
ENGINE_DRIVEN = {
    "--max-model-len": "utiliser plutôt le champ « max-model-len cible », qui adapte l'extension de contexte et les contrôles",
    "--max-num-seqs": "l'analyse SLO et la capture des CUDA graphs restent calculées sur la valeur générée",
    "--max-num-batched-tokens": "l'analyse SLO et le seuil de long prefill restent calculés sur la valeur générée",
    "--speculative-config": "les indicateurs de budget (d, a) restent calculés sur la config générée",
    "--tensor-parallel-size": "le KV et l'analyse SLO restent calculés sur le nombre de GPU par instance du formulaire",
}


def normalize_flag(flag: str) -> str:
    flag = flag.strip()
    return flag if flag.startswith("--") else "--" + flag.lstrip("-")


def decide_extra_args(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    req = ctx.request
    known = set(ctx.extracted.cli_flags)
    generated = {spec.cli: key for key, spec in ctx.vllm.flags.items()}
    overridden: dict[str, str] = {}
    tag = ctx.vllm.source_tag

    for raw, value in req.extra_args.items():
        flag = normalize_flag(raw)
        negated = flag.startswith("--no-") and flag not in known and "--" + flag[5:] in known
        if flag not in known and not negated:
            out.warnings.append(Warning(
                severity="critical", category="extra_args", rule_id="R-EXTRA-01",
                message=(f"Argument supplémentaire {flag} absent de vLLM {ctx.vllm.version} (code source au tag {tag}) :"
                         " ignoré. Le frontend Rust de vLLM ≥ 0.33 refuse de démarrer sur un flag inconnu."),
            ))
            continue
        if value is None or value is False:
            out.warnings.append(Warning(
                severity="warning", category="extra_args", rule_id="R-EXTRA-01",
                message=f"{flag} : valeur {value!r} non émise. Pour désactiver un booléen, écrire --no-<flag>: true.",
            ))
            continue
        if value is True:
            kind = "store_true"
        elif isinstance(value, (dict, list)):
            kind = "json"
        else:
            kind = "value"
            if isinstance(value, str) and value.strip().startswith(("{", "[")):
                try:
                    value, kind = json.loads(value), "json"
                except json.JSONDecodeError:
                    out.warnings.append(Warning(
                        severity="warning", category="extra_args", rule_id="R-EXTRA-01",
                        message=f"{flag} : la valeur ressemble à du JSON mais ne se lit pas ; émise telle quelle.",
                    ))
        if negated:
            out.warnings.append(Warning(
                severity="warning", category="to_verify", rule_id="R-EXTRA-01",
                message=f"À vérifier : la forme {flag} n'est pas listée telle quelle (seul --{flag[5:]} l'est) ; la confirmer sur l'image.",
            ))
        base_key = generated.get(flag) or generated.get("--" + flag[5:] if negated else "")
        note = ENGINE_DRIVEN.get(flag)
        justification = "Argument supplémentaire fourni en entrée."
        if base_key:
            overridden[base_key] = flag
            justification += f" Remplace la valeur générée pour {ctx.vllm.flags[base_key].cli}."
        if note:
            justification += f" Attention : {note}."
            out.warnings.append(Warning(
                severity="warning", category="extra_args", rule_id="R-EXTRA-01",
                message=f"{flag} forcé en argument supplémentaire : {note}.",
            ))
        out.decisions.append(Decision(
            key=f"extra:{flag}", value=value, rule_id="R-EXTRA-01", title=f"Argument supplémentaire {flag}",
            justification=justification, sources=[f"catalog/vllm/extracted/{ctx.vllm.version}.yaml (tag {tag})"],
            confidence="unverified", cli=flag, cli_kind=kind,
        ))

    env = dict(state.get("env", {}))
    for name, value in req.extra_env.items():
        if name in env and env[name] != value:
            out.warnings.append(Warning(
                severity="warning", category="extra_args", rule_id="R-EXTRA-02",
                message=f"Variable {name} : la valeur fournie ({value!r}) remplace celle de l'app ({env[name]!r}).",
            ))
        env[name] = str(value)
    out.state.update({"env": env, "overridden_by_extra": overridden})
    return out
