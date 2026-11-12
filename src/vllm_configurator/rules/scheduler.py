"""Scheduler rules: token budget, max_num_seqs, long prefill threshold, chunked prefill."""

from __future__ import annotations

import math
from typing import Any

from .. import formulas as f
from ..domain import Assumption, Decision, Indicator, Warning
from .context import KB, RuleContext, RuleOutput, fmt_int


def decide_batched_tokens(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    wl = ctx.request.workload
    profile = ctx.profile
    budget = profile.recommended_budget
    reason = (
        f"Profil « {profile.label} » : budget conseillé {fmt_int(budget)} "
        f"(plage {fmt_int(profile.budget_range[0])}–{fmt_int(profile.budget_range[1])}, {profile.source})."
    )
    if wl.mode == "batch" and budget < 16384:
        budget = 16384
        reason += " Mode batch : au moins 16 384, un budget élevé n'est rentable qu'avec beaucoup de prefill non caché (§4)."
    elif wl.mode == "interactive":
        reason += (
            " En interactif, le budget arbitre entre TTFT (budget haut) et ITL (budget bas) : un step chargé en prefill"
            " allonge l'ITL de toutes les séquences en decode (§2.2, §4)."
        )
    out = RuleOutput(state={"max_num_batched_tokens": budget})
    out.decisions.append(Decision(
        key="max_num_batched_tokens", value=budget, rule_id="R-SCHED-01", title="Budget de tokens par step",
        justification=reason + " Le remplissage réel est vérifié par l'indicateur « ratio ISL_eff/OSL ».",
        formula="Interactif : 8192 ; batch, long contexte ou sortie courte : 16384+ (§5)",
        sources=[f"{KB} §2.2, §4, §5"], default=ctx.default("max_num_batched_tokens"),
    ))
    return out


def decide_max_num_seqs(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    wl = ctx.request.workload
    kv_tokens: int | None = state.get("kv_tokens")
    start = ctx.heuristic("max_num_seqs_start")
    large_min = ctx.heuristic("large_kv_min_concurrency").value
    steps: list[str] = []

    kv_conc = math.floor(f.kv_concurrency(kv_tokens, wl.isl.p50)) if kv_tokens else None
    if kv_conc is not None and kv_conc >= large_min:
        seqs = start.value["large_kv"]
        steps.append(
            f"KV large : {fmt_int(kv_conc)} contextes P50 ≥ {fmt_int(large_min)} → point de départ {seqs} (§5)."
        )
    else:
        seqs = start.value["default"]
        steps.append(f"Point de départ {seqs} (§5 : 256–512 selon la taille du KV).")

    if wl.target_concurrency:
        per_instance = math.ceil(wl.target_concurrency / ctx.request.num_instances)
        if per_instance < seqs:
            seqs = per_instance
            steps.append(
                f"Concurrence cible {fmt_int(wl.target_concurrency)} sur {ctx.request.num_instances} instance(s)"
                f" → {per_instance} par instance : inutile d'aller au-delà (loi de Little)."
            )
        else:
            steps.append(
                f"Concurrence cible par instance ({per_instance}) ≥ {seqs} : les requêtes au-delà attendent en file."
                " Ajouter des instances ou relever max_num_seqs après mesure de l'ITL."
            )
            out.warnings.append(Warning(
                severity="warning", category="itl", rule_id="R-SCHED-02",
                message=(
                    f"Concurrence cible par instance ({per_instance}) supérieure à max_num_seqs ({seqs}) : file"
                    " d'attente, donc TTFT dégradé. Relever max_num_seqs uniquement si l'ITL p95 reste sous la cible."
                ),
            ))
    else:
        out.assumptions.append(Assumption(
            key="target_concurrency", label="Concurrence cible", value="non fournie",
            reason="max_num_seqs suit la capacité du KV et le point de départ §5 ; la limite ITL reste à mesurer.",
        ))

    if kv_conc is not None and kv_conc < seqs:
        seqs = max(1, kv_conc)
        steps.append(f"Plafonné par le KV : {fmt_int(kv_conc)} contextes P50 seulement.")

    out.state["max_num_seqs"] = seqs
    formula = "min(concurrence_KV au contexte P50, limite ITL) (§5)"
    computation = (
        f"concurrence_KV = {fmt_int(kv_tokens)} / {fmt_int(wl.isl.p50)} = {fmt_int(kv_conc)}" if kv_conc is not None
        else "concurrence_KV inconnue"
    )
    out.decisions.append(Decision(
        key="max_num_seqs", value=seqs, rule_id="R-SCHED-02", title="Séquences actives maximales",
        justification=" ".join(steps) + " Le trafic agentique est surtout du decode : la capacité du KV n'est pas le facteur limitant.",
        formula=formula, computation=computation, confidence=start.status,
        sources=[f"{KB} §2.3, §3.2, §5"], default=ctx.default("max_num_seqs"),
    ))
    return out


def decide_long_prefill(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    budget = state["max_num_batched_tokens"]
    divisor = ctx.heuristic("long_prefill_threshold_divisor")
    bypass = ctx.behavior("long_prefill.single_request_bypass")
    adaptive = ctx.behavior("long_prefill.adaptive_floor")
    threshold = budget // divisor.value

    justification = (
        f"Équité entre un gros prefill (reprise de session) et les petits tours : jusqu'à ~{divisor.value} prefills"
        f" en parallèle en cas de contention."
    )
    if bypass.value:
        justification += " Depuis 0.31, une requête seule n'est pas plafonnée (#57951) : un seuil bas ne pénalise pas les prompts isolés."
    else:
        out.warnings.append(Warning(
            severity="warning", category="version", rule_id="R-SCHED-03",
            message="Cette version plafonne même une requête seule : un seuil bas gaspille du budget à faible charge.",
            source=bypass.source,
        ))
    out.decisions.append(Decision(
        key="long_prefill_token_threshold", value=threshold, rule_id="R-SCHED-03", title="Seuil de long prefill",
        justification=justification, formula="seuil ≈ max_num_batched_tokens / 4 (§2.5, §5)",
        computation=f"{fmt_int(budget)} / {divisor.value} = {fmt_int(threshold)}", confidence=divisor.status,
        sources=[f"{KB} §2.5", bypass.source], default=ctx.default("long_prefill_token_threshold"),
    ))

    if adaptive.value == "opt_in" and ctx.has_flag("long_prefill_token_threshold_adaptive"):
        seqs = state["max_num_seqs"]
        loaded_floor = budget // (seqs + 1)
        out.decisions.append(Decision(
            key="long_prefill_token_threshold_adaptive", value=True, rule_id="R-SCHED-04",
            title="Plancher adaptatif du seuil de long prefill",
            justification=(
                "Sans ce flag (désactivé par défaut), le comportement dynamique de #58459 décrit au §2.5 ne s'applique pas."
                " Il relève le seuil au partage équitable du budget quand peu de requêtes sont présentes"
                " (exemple de la PR : 2 requêtes, budget 8192 → 4096 chacune au lieu de 2048)."
                " Sous forte charge, les séquences en decode entrent dans le diviseur : le plancher devient négligeable"
                f" ({fmt_int(budget)} // {seqs + 1} = {loaded_floor} avec {seqs} séquences en decode et 1 prefill)"
                f" et c'est le seuil fixe de {fmt_int(threshold)} qui s'applique."
            ),
            formula="seuil_effectif = max(seuil, max_num_batched_tokens // (running + waiting))",
            computation=f"1 prefill + 1 decode : max({threshold}, {budget} // 2) = {max(threshold, budget // 2)}",
            sources=[adaptive.source], default=ctx.default("long_prefill_token_threshold_adaptive"),
        ))
    return out


def decide_chunked_prefill(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    out.decisions.append(Decision(
        key="enable_chunked_prefill", value=True, rule_id="R-SCHED-05", title="Chunked prefill",
        justification=(
            "Le remplissage du budget (decode d'abord, puis morceaux de prefill) et le seuil de long prefill reposent"
            " sur le chunked prefill. C'est le défaut ; il est émis explicitement pour figer la config."
        ),
        sources=[f"{KB} §2.2", ctx.vllm.defaults["enable_chunked_prefill"].source],
        default=ctx.default("enable_chunked_prefill"),
    ))
    return out


def indicate_scheduler(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    """§3.1 and §3.4 indicators, computed once every scheduler and spec decision is known."""
    out = RuleOutput()
    wl = ctx.request.workload
    budget, seqs = state["max_num_batched_tokens"], state["max_num_seqs"]
    k, d, a = state["num_speculative_tokens"], state["decode_tokens_per_seq"], state["acceptance_length"]

    active = seqs
    if wl.target_concurrency:
        active = min(seqs, math.ceil(wl.target_concurrency / ctx.request.num_instances))
    else:
        out.assumptions.append(Assumption(
            key="sustained_concurrency", label="Concurrence soutenue", value=f"≥ {seqs}",
            reason="Hypothèse du §3.1 : le budget ne se remplit en régime stable qu'avec une concurrence soutenue ≥ S (loi de Little).",
        ))

    needed = f.required_ratio(budget, active, d, a)
    out.indicators.append(Indicator(
        key="required_ratio", label="Ratio ISL_eff/OSL nécessaire pour remplir le budget", value=round(needed, 1),
        formula="(B − S·d) / (S·a) (§3.1)",
        computation=f"({fmt_int(budget)} − {active}·{d}) / ({active}·{a:g}) = {needed:.1f}",
        interpretation=(
            f"d = {d} : en decode, chaque séquence consomme 1 token" + (f" + {k} tokens de vérification" if k else "")
            + (f" + {d - 1 - k} slots de drafting" if d - 1 - k > 0 else "") + " dans le budget."
        ),
    ))

    hit = state["prefix_cache_hit_rate"]
    eff = f.effective_ratio(wl.isl.p50, hit, wl.osl.p50)
    fills = eff >= needed
    out.indicators.append(Indicator(
        key="effective_ratio", label="Ratio ISL_eff/OSL du profil fourni", value=round(eff, 2),
        formula="ISL_P50 × (1 − hit rate) / OSL_P50",
        computation=f"{fmt_int(wl.isl.p50)} × (1 − {hit:.3f}) / {fmt_int(wl.osl.p50)} = {eff:.2f}",
        interpretation=(
            "≥ ratio nécessaire : le budget peut être rempli en régime stable (steps longs, ITL à surveiller)."
            if fills else
            "< ratio nécessaire : le budget n'est rempli que lors des rafales de prefill ; le régime stable est du decode."
        ),
    ))
    ref = ctx.profile.reference_ratio
    if ref is not None:
        ref_fills = ref.value >= needed
        out.indicators.append(Indicator(
            key="reference_ratio", label="Ratio effectif publié pour ce type d'usage", value=ref.value,
            computation=ref.source, confidence=ref.status,
            interpretation=(
                ("≥" if ref_fills else "<") + f" ratio nécessaire ({needed:.1f}) : avec un trafic réel de ce type, le budget "
                + ("peut être rempli à cette concurrence." if ref_fills else "n'est rempli que lors des rafales.")
            ),
        ))

    verified = f.verified_tokens_per_step(active, k)
    out.indicators.append(Indicator(
        key="decode_tokens_per_step", label="Tokens vérifiés par step de decode pur", value=verified,
        formula="S × (k + 1) (§3.4)", computation=f"{active} × {k + 1} = {fmt_int(verified)}",
        interpretation=f"{verified / budget:.0%} du budget consommé par le decode avant tout prefill.",
    ))
    if verified >= budget:
        out.warnings.append(Warning(
            severity="critical", category="itl", rule_id="R-SCHED-02",
            message=(
                f"Le decode seul ({fmt_int(verified)} tokens par step) remplit le budget de {fmt_int(budget)} tokens :"
                " plus aucune place pour le prefill. Augmenter le budget ou réduire max_num_seqs."
            ),
        ))

    moe = ctx.model.architecture.moe
    if moe is not None:
        interpretations = {
            "decode": "Step de decode : peu de tokens par expert, GEMM des experts memory-bound (§2.2).",
            "full": f"Step plein : ×{budget / verified:.1f} plus de tokens par expert qu'en decode pur ;"
                    " ce sont ces steps chargés en prefill qui rendent les experts efficaces (§2.2).",
        }
        for key, label, tokens in (("decode", "step de decode pur", verified), ("full", "step plein", budget)):
            value = f.tokens_per_expert(tokens, moe.num_experts_per_tok, moe.n_routed_experts)
            out.indicators.append(Indicator(
                key=f"tokens_per_expert_{key}", label=f"Tokens par expert MoE ({label})", value=round(value, 1),
                formula="T × top_k / n_experts (§3.6)",
                computation=f"{fmt_int(tokens)} × {moe.num_experts_per_tok} / {moe.n_routed_experts} = {value:.1f}",
                interpretation=interpretations[key], confidence=ctx.model.architecture.status,
            ))
    return out
