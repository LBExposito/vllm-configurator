"""Speculative decoding rules (method, k, FLy, budget accounting)."""

from __future__ import annotations

from typing import Any

from .. import formulas as f
from ..domain import Assumption, Decision, Indicator, Warning
from .context import KB, RuleContext, RuleOutput, unverified_warning

LOSSY_USAGES = {"agentic"}


def _disabled(reason: str, ctx: RuleContext) -> RuleOutput:
    out = RuleOutput(state={"num_speculative_tokens": 0, "decode_tokens_per_seq": 1, "acceptance_length": 1.0,
                            "spec_method": None})
    out.decisions.append(Decision(
        key="speculative_config", value=None, rule_id="R-SPEC-01", title="Speculative decoding",
        justification=reason, emitted=False, omitted_reason=reason, sources=[f"{KB} §2.4"],
    ))
    return out


def decide_speculative(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    req, spec_model = ctx.request, ctx.model.speculative
    opts = req.spec
    if not opts.enabled:
        return _disabled("Désactivé à la demande.", ctx)
    method = spec_model.native_method
    if method is None:
        return _disabled("Le modèle n'a pas de méthode de speculative decoding native au catalogue.", ctx)

    out = RuleOutput()
    if method not in ctx.extracted.speculative_methods:
        out.warnings.append(Warning(
            severity="critical", category="unsupported", rule_id="R-SPEC-01",
            message=f"La méthode {method!r} n'existe pas dans vLLM {ctx.vllm.version} : speculative decoding désactivé.",
            source=ctx.extracted.source.get("tag"),
        ))
        disabled = _disabled(f"Méthode {method!r} absente de vLLM {ctx.vllm.version}.", ctx)
        disabled.warnings = out.warnings
        return disabled

    recommended = spec_model.recommended_num_speculative_tokens
    if spec_model.fixed_k:
        k = recommended
        if opts.num_speculative_tokens and opts.num_speculative_tokens != k:
            out.warnings.append(Warning(
                severity="warning", category="model_note", rule_id="R-SPEC-01",
                message=f"k = {opts.num_speculative_tokens} demandé, mais le checkpoint impose k = {k} : valeur imposée retenue.",
                source=", ".join(spec_model.source) if isinstance(spec_model.source, list) else spec_model.source,
            ))
        k_reason = f"k = {k}, imposé par le checkpoint."
    elif opts.num_speculative_tokens:
        k = opts.num_speculative_tokens
        k_reason = f"k = {k}, fourni en entrée" + (f" (la recette recommande {recommended})." if recommended and recommended != k else ".")
    elif recommended:
        k = recommended
        k_reason = f"k = {k}, recommandé pour ce modèle."
    else:
        return _disabled("Aucun k recommandé au catalogue et aucun k fourni : à renseigner.", ctx)

    slots_table = ctx.behavior("spec_decode.extra_draft_slots")
    expr = slots_table.value.get(method)
    if expr is None:
        expr = "0"
        out.warnings.append(unverified_warning(
            f"réservation de slots de drafting pour la méthode {method!r} (supposée nulle)", slots_table.source, "R-SPEC-02"))
    extra = f.eval_slot_expression(expr, k)
    in_budget = bool(ctx.behavior("spec_decode.verification_tokens_in_budget").value)
    d = f.decode_tokens_per_seq(k, extra, in_budget)

    if opts.acceptance_length:
        a = opts.acceptance_length
        a_source = "mesure fournie"
    else:
        h = ctx.heuristic("assumed_acceptance_length")
        a = min(float(h.value), k + 1.0)
        a_source = "hypothèse"
        out.assumptions.append(Assumption(
            key="acceptance_length", label="Longueur d'acceptation moyenne a", value=a,
            reason=f"{h.source}. {h.note}", status=h.status,
        ))

    config: dict[str, Any] = {"method": method, "num_speculative_tokens": k}
    draft_sample = opts.draft_sample_method or ctx.default("draft_sample_method")
    config["draft_sample_method"] = draft_sample

    fly = ctx.behavior("spec_decode.fly").value
    usage = req.workload.usage_type
    lossy_reason = "FLy (vérification lossy) non demandé : rejection sampling standard, sortie identique au modèle cible."
    if req.allow_lossy and fly.get("available") and k >= fly["min_k"]:
        window = min(6, k - 1)
        config.update({
            "rejection_sample_method": "fly", "fly_window_size": window,
            "fly_entropy_threshold": fly["default_entropy_threshold"], "fly_entropy_top_k": fly["default_entropy_top_k"],
        })
        lossy_reason = (
            f"FLy activé (tolérance lossy acceptée) : fenêtre {window} = min(6, k − 1), défauts vLLM pour l'entropie."
            " Gain annoncé ×1,04 à ×1,31 sur le decode ; qualité conservée ≥ ~97,9 % au pire (§2.4)."
        )
        out.warnings.append(Warning(
            severity="critical" if usage in LOSSY_USAGES or ctx.profile.tool_calls else "warning", category="lossy",
            rule_id="R-SPEC-03",
            message=(
                "FLy est lossy : il peut modifier la sortie du modèle."
                + (" À éviter par défaut en agentique (JSON des tool calls, arguments exacts) : n'activer qu'en A/B"
                   " en mesurant le taux de tool calls valides." if usage in LOSSY_USAGES or ctx.profile.tool_calls else "")
            ),
            source=f"{KB} §2.4 ; vllm/config/speculative.py (avertissement « lossy speculative decoding method »)",
        ))
    else:
        if req.allow_lossy and not fly.get("available"):
            lossy_reason = f"FLy indisponible dans vLLM {ctx.vllm.version} (arrive en 0.33, #53987)."
        elif req.allow_lossy:
            lossy_reason = f"FLy exige k ≥ {fly['min_k']}."
        config["rejection_sample_method"] = ctx.default("rejection_sample_method")

    adaptive = ctx.behavior("spec_decode.adaptive_verification").value
    if adaptive.get("available") and method in adaptive.get("methods", []):
        config["enable_adaptive_verification"] = bool(adaptive.get("default", False))

    unknown = [key for key in config if key not in ctx.extracted.speculative_config_keys]
    if unknown:
        raise ValueError(f"clés de --speculative-config absentes de vLLM {ctx.vllm.version} : {unknown}")
    allowed = ctx.extracted.rejection_sample_methods
    if config["rejection_sample_method"] not in allowed:
        raise ValueError(f"rejection_sample_method {config['rejection_sample_method']!r} absent de {allowed}")

    sources = [f"{KB} §2.4"] + ([spec_model.source] if isinstance(spec_model.source, str) else list(spec_model.source))
    parts = [
        f"Méthode native du modèle : {method}.", k_reason, lossy_reason,
        f"draft_sample_method = {draft_sample}"
        + (" (échantillonne la distribution du drafter ; coûte de la mémoire GPU supplémentaire)." if draft_sample == "probabilistic" else " (défaut vLLM)."),
    ]
    if "enable_adaptive_verification" in config:
        parts.append(
            "enable_adaptive_verification = false (défaut vLLM, DSpark uniquement, vérifié dans le code source) :"
            " la vérification adaptative fait varier le nombre de tokens vérifiés par requête ; à évaluer en A/B."
        )
    out.decisions.append(Decision(
        key="speculative_config", value=config, rule_id="R-SPEC-01", title="Speculative decoding",
        justification=" ".join(parts), confidence=spec_model.status, sources=sources,
    ))
    out.indicators.append(Indicator(
        key="decode_tokens_per_seq", label="Tokens de budget consommés par séquence en decode (d)", value=d,
        formula="d = 1 + k (vérification) + slots de drafting supplémentaires",
        computation=f"1 + {k} + {extra} = {d} (slots {method} : {expr!r} en vLLM {ctx.vllm.version})",
        interpretation=(
            "Contrairement au §2.4 et au §3.1 de la base, les tokens de vérification restent comptés en 0.33 :"
            " #59468 ne supprime que les slots de drafting supplémentaires de DSpark/DFlash."
            if method in ("dspark", "dflash") and in_budget else None
        ),
    ))

    if ctx.behavior("spec_decode.dspark_active_seq_cap").value and method in ("dspark", "dflash"):
        budget, seqs = state["max_num_batched_tokens"], state["max_num_seqs"]
        cap = f.dspark_active_seq_cap(budget, k + 1)
        out.indicators.append(Indicator(
            key="dspark_active_seq_cap", label="Plafond vLLM de séquences running (DSpark/DFlash)", value=cap,
            formula="max_num_batched_tokens // num_lookahead_tokens (#59468), num_lookahead_tokens = k + 1 au pire",
            computation=f"{budget} // {k + 1} = {cap}",
            interpretation=("Ne contraint pas max_num_seqs." if cap >= seqs else
                            f"vLLM abaissera max_num_active_seqs à {cap} (< max_num_seqs = {seqs})."),
        ))
        if cap < seqs:
            out.warnings.append(Warning(
                severity="warning", category="version", rule_id="R-SPEC-04",
                message=f"vLLM plafonnera les séquences running à {cap} (budget {budget} // {k + 1}) : relever le budget ou baisser max_num_seqs.",
                source=ctx.behavior("spec_decode.dspark_active_seq_cap").source,
            ))

    lm_head_q = ctx.behavior("spec_decode.draft_lm_head_quantization").value
    if lm_head_q.get("available") and method in ("dspark", "dflash"):
        out.warnings.append(Warning(
            severity="info", category="version", rule_id="R-SPEC-05",
            message=(
                "Option disponible non activée : draft_lm_head_quantization (fp8 ou nvfp4), environ +18 % sur le decode"
                " à concurrence 1 (#59973). Impact sur l'acceptation non évalué ici : à tester en A/B."
            ),
            source=ctx.behavior("spec_decode.draft_lm_head_quantization").source,
        ))

    out.state.update({"num_speculative_tokens": k, "decode_tokens_per_seq": d, "acceptance_length": a,
                      "spec_method": method, "acceptance_source": a_source})
    return out
