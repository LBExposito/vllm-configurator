"""Parallelism, workload diagnostics, support checks and version notes."""

from __future__ import annotations

from typing import Any

from ..catalog.loader import version_key
from ..domain import Assumption, Decision, Warning
from .context import KB, RuleContext, RuleOutput, fmt_int, unverified_warning


def decide_parallelism(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    tp = ctx.request.gpus_per_instance
    out.decisions.append(Decision(
        key="tensor_parallel_size", value=tp, rule_id="R-PAR-01", title="Tensor parallel",
        justification=f"{tp} GPU par instance, en tensor parallel. Les instances sont indépendantes : le routage entre elles relève du load balancer (EPP llm-d en v2).",
        sources=[f"{KB} §1"], default=1,
    ))
    out.decisions.append(Decision(
        key="data_parallel_size", value=1, rule_id="R-PAR-02", title="Data parallel dans l'instance",
        justification=(
            "DP1 : max_num_seqs s'applique à toute l'instance. Le DP-attention n'est utile que si le KV est limitant,"
            " car il évite de répliquer le KV MLA entre rangs TP (§5)."
        ),
        sources=[f"{KB} §2.3, §5"], default=1,
    ))
    return out


def resolve_workload(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    wl, profile = ctx.request.workload, ctx.profile
    hit = wl.prefix_cache_hit_rate
    if hit is None:
        if profile.prefix_cache_hit_rate is not None:
            hit = profile.prefix_cache_hit_rate.value
            out.assumptions.append(Assumption(
                key="prefix_cache_hit_rate", label="Taux de prefix cache hit", value=hit,
                reason=f"Non fourni : valeur publiée pour « {profile.label} » ({profile.prefix_cache_hit_rate.source}).",
                status=profile.prefix_cache_hit_rate.status,
            ))
        else:
            hit = 0.0
            out.assumptions.append(Assumption(
                key="prefix_cache_hit_rate", label="Taux de prefix cache hit", value=0.0,
                reason=f"Non fourni et aucune valeur publiée pour « {profile.label} » : hypothèse prudente (tout le prompt est recalculé).",
            ))
    out.state["prefix_cache_hit_rate"] = hit

    ref = profile.osl_per_call_median
    factor = ctx.heuristic("osl_suspect_factor").value
    if ref is not None and wl.osl.p50 > factor * ref.value:
        out.warnings.append(Warning(
            severity="warning", category="osl_suspect", rule_id="R-DIAG-01",
            message=(
                f"OSL P50 = {fmt_int(wl.osl.p50)}, soit ×{wl.osl.p50 / ref.value:.0f} la médiane publiée par appel"
                f" ({fmt_int(ref.value)}). Causes probables : benchmark à sortie fixe (--random-output-len, ignore_eos),"
                " mesure par tour au lieu de par appel, ou thinking long. Rejouer des sessions multi-tours réelles."
            ),
            source=f"{ref.source} ; {KB} §6 ; seuil ×{factor} : {ctx.heuristic('osl_suspect_factor').source}",
        ))
    tail = ctx.heuristic("isl_tail_ratio_warning").value
    if wl.isl.p95 and wl.isl.p99 and wl.isl.p99 / wl.isl.p95 > tail:
        out.warnings.append(Warning(
            severity="info", category="isl_tail", rule_id="R-DIAG-02",
            message=(
                f"ISL P99 ({fmt_int(wl.isl.p99)}) = ×{wl.isl.p99 / wl.isl.p95:.1f} le P95 : vérifier s'il s'agit d'une mesure"
                " par appel ou d'un cumul par tour ou par session. Sans compactage du contexte, c'est possible avec un contexte de 1M."
            ),
            source=f"{KB} §6",
        ))
    return out


def check_support(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    model, ver = ctx.model, ctx.vllm.version
    serving = model.serving
    if model.architecture.hf_architecture not in ctx.extracted.architectures:
        out.warnings.append(Warning(
            severity="critical", category="unsupported", rule_id="R-SUP-01",
            message=(
                f"L'architecture {model.architecture.hf_architecture} est absente du registre de vLLM {ver}."
                + (f" Image dédiée requise : {serving.requires_custom_image}." if serving.requires_custom_image else "")
                + " Vérifier les flags sur l'image réellement déployée."
            ),
            source=f"vllm/model_executor/models/registry.py au tag {ctx.vllm.source_tag}",
        ))
    if serving.min_vllm_version and version_key(ver) < version_key(serving.min_vllm_version):
        out.warnings.append(Warning(
            severity="critical", category="unsupported", rule_id="R-SUP-02",
            message=f"Version minimale pour ce modèle : {serving.min_vllm_version} (demandée : {ver}).",
            source=serving.source if isinstance(serving.source, str) else " ; ".join(serving.source),
        ))
    if serving.status == "unverified":
        out.warnings.append(unverified_warning("prise en charge du modèle par cette version de vLLM", serving.source,
                                               "R-SUP-03", serving.note))
    for note in model.notes:
        out.warnings.append(Warning(severity="info", category="model_note", message=note, rule_id="R-SUP-04"))
    return out


def version_notes(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    for item in ctx.vllm.breaking_changes:
        out.warnings.append(Warning(
            severity="info", category="breaking_change", rule_id="R-VER-01",
            message=f"Changement cassant en {ctx.vllm.version} ({item.id}) : {item.text}",
            source="Notes de version vLLM" if item.status == "verified" else "À vérifier",
        ))
    for key, behavior in ctx.vllm.behaviors.items():
        if behavior.status == "unverified":
            out.warnings.append(unverified_warning(f"comportement vLLM « {key} » ({behavior.value})", behavior.source,
                                                   "R-VER-02", behavior.note))
    return out


def itl_watch(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    seqs, k = state["max_num_seqs"], state["num_speculative_tokens"]
    fallback = max(1, seqs * 3 // 4)
    target = ctx.request.slo.itl_p95_ms
    out.warnings.append(Warning(
        severity="info", category="itl", rule_id="R-DIAG-03",
        message=(
            f"À surveiller après déploiement : ITL p95" + (f" (cible {target:g} ms)" if target else "")
            + f". Si elle dépasse la cible, redescendre à {fallback} séquences et à une capture de {fallback * (k + 1)}."
            " Vérifier aussi l'absence d'OOM pendant la capture des CUDA graphs."
        ),
        source=f"{KB} §7",
    ))
    if ctx.request.slo.ttft_p95_ms:
        out.warnings.append(Warning(
            severity="info", category="itl", rule_id="R-DIAG-04",
            message=(
                f"TTFT p95 cible {ctx.request.slo.ttft_p95_ms:g} ms : aucun modèle de latence n'est calibré dans l'app."
                " Valider avec le test du seuil de long prefill (onglet Vérification)."
            ),
        ))
    return out
