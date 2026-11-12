"""KV cache, memory and CUDA graph rules."""

from __future__ import annotations

import math
from typing import Any

from .. import formulas as f
from ..domain import Assumption, Decision, Indicator, Warning
from .context import KB, RuleContext, RuleOutput, fmt_int, unverified_warning

GIB = 1024**3


def resolve_kv_capacity(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    """KV tokens per instance: startup log first, then catalog estimate, else blocking."""
    out = RuleOutput()
    req = ctx.request
    if req.observed_kv is not None:
        tokens = req.observed_kv.kv_cache_tokens
        out.state.update({"kv_tokens": tokens, "kv_source": "startup_log"})
        out.indicators.append(Indicator(
            key="kv_tokens", label="KV cache par instance (log de démarrage)", value=tokens, unit="tokens",
            computation="« GPU KV cache size » lu dans les logs vLLM",
        ))
        return out

    kv, weights = ctx.model.kv_cache, ctx.model.weights
    missing = []
    if kv.bytes_per_token is None:
        missing.append(f"octets de KV par token du modèle ({kv.method.strip()})")
    if kv.replicated_across_tp is None:
        missing.append("répartition du KV entre rangs TP")
    if weights.checkpoint_bytes is None:
        missing.append("taille des poids")
    if missing:
        out.state.update({"kv_tokens": None, "kv_source": "unknown",
                          "blocking": ["Taille du KV inconnue : fournissez la valeur « GPU KV cache size » des logs de démarrage"
                                       " vLLM (une instance). Données manquantes pour l'estimer : " + " ; ".join(missing)]})
        return out

    util = state["gpu_memory_utilization_hint"]
    reserve_gib = req.kv_estimate_reserve_gib_per_gpu
    if reserve_gib is None:
        reserve_gib = 0.0
        out.assumptions.append(Assumption(
            key="kv_reserve", label="Réserve mémoire par GPU (activations, CUDA graphs, drafter)", value="0 Gio",
            reason="Non fournie : l'estimation du KV est une borne haute.",
        ))
    tp = req.gpus_per_instance
    tokens = f.kv_tokens_estimate(
        hbm_bytes_per_gpu=ctx.gpu.memory.hbm_gb * 1e9, gpu_memory_utilization=util,
        weights_bytes=weights.checkpoint_bytes, tensor_parallel_size=tp, reserve_bytes_per_gpu=reserve_gib * GIB,
        bytes_per_token=kv.bytes_per_token, replicated_across_tp=kv.replicated_across_tp,
        num_kv_heads=ctx.model.architecture.num_key_value_heads,
    )
    out.state.update({"kv_tokens": tokens, "kv_source": "estimate"})
    out.indicators.append(Indicator(
        key="kv_tokens", label="KV cache par instance (estimation)", value=tokens, unit="tokens",
        formula="(HBM × util − poids / TP − réserve) / octets_KV_par_token_par_GPU",
        computation=(
            f"({ctx.gpu.memory.hbm_gb:g} Go × {util:g} − {weights.checkpoint_bytes / 1e9:.0f} Go / {tp} − {reserve_gib:g} Gio)"
            f" / {kv.bytes_per_token:g} o" + ("" if kv.replicated_across_tp else " ÷ répartition TP") + f" = {fmt_int(tokens)}"
        ),
        interpretation="Estimation grossière : à remplacer par la valeur des logs de démarrage dès le premier déploiement.",
        confidence="unverified",
    ))
    out.warnings.append(unverified_warning(
        f"octets de KV par token ({kv.bytes_per_token:g} o en {kv.bytes_per_token_dtype})", kv.source, "R-KV-01", kv.method.strip()))
    if ctx.gpu.memory.status == "unverified":
        out.warnings.append(unverified_warning(
            f"capacité HBM du {ctx.gpu.display_name} ({ctx.gpu.memory.hbm_gb:g} Go)", ctx.gpu.memory.source, "R-KV-01",
            ctx.gpu.memory.note))
    if kv.per_sequence_state_bytes:
        out.warnings.append(Warning(
            severity="warning", category="kv_limited", rule_id="R-KV-01",
            message=(
                f"Le modèle garde aussi un état fixe de {kv.per_sequence_state_bytes / 1e6:.0f} Mo par séquence"
                f" (attention linéaire), réparti sur les rangs TP et non compté dans cette estimation : la concurrence"
                " réelle sera plus basse."
            ),
            source=kv.source if isinstance(kv.source, str) else " ; ".join(kv.source),
        ))
    return out


def decide_prefix_caching(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    out.decisions.append(Decision(
        key="enable_prefix_caching", value=True, rule_id="R-KV-02", title="Prefix caching",
        justification=(
            "Indispensable en agentique (95 à 98 % de hit mesurés, §6). System prompt et définitions d'outils doivent"
            " être identiques à l'octet près et placés en tête du prompt. C'est le défaut ; émis explicitement."
        ),
        sources=[f"{KB} §2.6", ctx.vllm.defaults["enable_prefix_caching"].source],
        default=ctx.default("enable_prefix_caching"),
    ))
    if any(kind.startswith("linear") for kind in ctx.model.architecture.attention):
        out.warnings.append(Warning(
            severity="warning", category="to_verify", rule_id="R-KV-02",
            message=(
                "À vérifier : modèle hybride avec attention linéaire. Le prefix caching de l'état linéaire dépend du mode"
                " de cache Mamba de vLLM ; mesurer le hit rate réel (vllm:prefix_cache_hits / queries)."
            ),
        ))
    return out


def decide_kv_dtype(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    kv = ctx.model.kv_cache
    dtype = kv.recommended_dtype or ctx.default("kv_cache_dtype")
    out = RuleOutput()
    out.decisions.append(Decision(
        key="kv_cache_dtype", value=dtype, rule_id="R-KV-03", title="Type du KV cache",
        justification=(
            f"Valeur recommandée pour ce modèle : {dtype}."
            + (" La disposition fp8_ds_mla des modèles DeepSeek exige fp8 (§5)." if ctx.model.architecture.model_type.startswith("deepseek") else "")
            + (" « auto » laisse vLLM choisir la disposition native du modèle." if dtype == "auto" else "")
        ),
        confidence=kv.recommended_dtype_status,
        sources=[s for s in (kv.recommended_dtype_source, f"{KB} §5") if s],
        default=ctx.default("kv_cache_dtype"),
    ))
    if kv.recommended_dtype_status == "unverified":
        out.warnings.append(unverified_warning(f"dtype de KV recommandé ({dtype})", kv.recommended_dtype_source or "—", "R-KV-03"))
    return out


def default_capture_ceiling(ctx: RuleContext) -> int:
    ceilings = ctx.behavior("cudagraph.default_max_capture_size").value
    return ceilings.get(ctx.gpu.cudagraph_family, ceilings["other"])


def decide_compilation(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    k, seqs, budget = state["num_speculative_tokens"], state["max_num_seqs"], state["max_num_batched_tokens"]
    raw = f.cudagraph_capture_size(k, seqs)
    size = raw
    notes = []
    if ctx.behavior("cudagraph.capped_by_max_num_batched_tokens").value or ctx.behavior("cudagraph.multiple_of_k_plus_1").value:
        size = f.clamp_capture_size(raw, budget, k + 1)
        if size != raw:
            notes.append(f"Ramené à {fmt_int(size)} : vLLM plafonne la capture à max_num_batched_tokens et l'arrondit à un multiple de k + 1.")
    ceiling = default_capture_ceiling(ctx)
    vllm_default = min(seqs * (k + 1) * 2, ceiling, budget)
    out.state["max_cudagraph_capture_size"] = size
    out.decisions.append(Decision(
        key="compilation_config", value={"max_cudagraph_capture_size": size}, rule_id="R-CG-01",
        title="Taille maximale de capture des CUDA graphs",
        justification=(
            "Un step de decode pur traite (k + 1) tokens par séquence : la capture doit couvrir S × (k + 1) pour que le"
            f" decode à pleine concurrence reste en CUDA graph. Sans ce réglage, vLLM plafonne la capture à {fmt_int(ceiling)}"
            f" sur ce GPU, soit {fmt_int(vllm_default)} ici." + (" " + " ".join(notes) if notes else "")
        ),
        formula="max_cudagraph_capture_size = (k + 1) × max_num_seqs (§3.3)",
        computation=f"({k} + 1) × {seqs} = {fmt_int(raw)}",
        sources=[f"{KB} §2.4, §3.3", ctx.behavior("cudagraph.default_max_capture_size").source],
        default=vllm_default,
    ))
    return out


def decide_gpu_memory_utilization(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    req = ctx.request
    default = ctx.default("gpu_memory_utilization")
    k = state["num_speculative_tokens"]
    size = state["max_cudagraph_capture_size"]
    ceiling = default_capture_ceiling(ctx)
    h = ctx.heuristic("gpu_memory_utilization_large_capture")
    if req.gpu_memory_utilization is not None:
        value, reason, confidence = req.gpu_memory_utilization, "Valeur fournie en entrée.", "verified"
    elif k > 0 and size > ceiling:
        value = h.value
        reason = (
            f"Capture de {fmt_int(size)} tokens avec speculative decoding, au-delà du plafond par défaut de vLLM"
            f" ({fmt_int(ceiling)}) : risque d'OOM pendant la capture. Recommandation du blog vLLM : {h.value}."
            " À relever si aucun OOM n'apparaît et si le KV devient limitant."
        )
        confidence = h.status
    else:
        value, reason, confidence = default, "Défaut de vLLM : aucune grosse capture de CUDA graphs à protéger.", "verified"
    out.state["gpu_memory_utilization"] = value
    out.decisions.append(Decision(
        key="gpu_memory_utilization", value=value, rule_id="R-KV-04", title="Fraction de mémoire GPU",
        justification=reason, formula="0,8 avec de grosses captures de CUDA graphs et spec decoding (§5)",
        confidence=confidence, sources=[f"{KB} §2.4, §5", h.source], default=default,
    ))
    observed = req.observed_kv
    if observed is not None and observed.gpu_memory_utilization != value:
        measured = f"{observed.gpu_memory_utilization:g}" if observed.gpu_memory_utilization else "inconnue"
        out.warnings.append(Warning(
            severity="warning", category="memory", rule_id="R-KV-04",
            message=(
                f"Le KV des logs ({fmt_int(observed.kv_cache_tokens)} tokens) a été mesuré avec gpu-memory-utilization = {measured},"
                f" la config retient {value:g}. La taille du KV changera aussi avec max-num-seqs et la capture des CUDA graphs :"
                " relire « GPU KV cache size » après le déploiement (et recalculer l'EPP llm-d en v2)."
            ),
        ))
    return out


def decide_block_size(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    bs = ctx.model.block_size
    observed = ctx.request.observed_kv.block_size if ctx.request.observed_kv else None
    if bs.forced_by_backend:
        reason = f"Imposée à {bs.forced_by_backend} par le backend d'attention : ne pas forcer --block-size."
        value, source = bs.forced_by_backend, "backend_forced"
    elif observed:
        reason = f"Taille lue dans les logs ({observed}) : laissée au backend, pas de --block-size."
        value, source = observed, "startup_log"
    else:
        reason = "Laissée au backend. Lire « kv cache block size » dans les logs de démarrage (utile pour l'EPP llm-d en v2)."
        value, source = None, "unknown"
        if bs.status == "unverified":
            out.warnings.append(unverified_warning("taille de bloc KV réelle", bs.source, "R-KV-05", bs.note))
    out.state.update({"block_size": value, "block_size_source": source})
    out.decisions.append(Decision(
        key="block_size", value=value, rule_id="R-KV-05", title="Taille de bloc KV", justification=reason,
        confidence=bs.status, sources=[bs.source] if isinstance(bs.source, str) else list(bs.source),
        emitted=False,
        omitted_reason=(f"imposée à {value} par le backend, ne pas la forcer (§5)" if source == "backend_forced"
                        else "laissée au backend, ne pas la forcer (§5)"),
    ))
    return out


def decide_kv_offload(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    wl = ctx.request.workload
    kv_tokens = state.get("kv_tokens")
    conc = math.floor(f.kv_concurrency(kv_tokens, wl.isl.p50)) if kv_tokens else None
    sessions = None
    if wl.target_concurrency is not None:
        sessions = math.ceil((wl.target_concurrency + (wl.paused_sessions or 0)) / ctx.request.num_instances)
    if conc is not None and sessions is not None and conc < sessions:
        reason = (
            f"Le KV loge {fmt_int(conc)} contextes P50 pour {fmt_int(sessions)} sessions actives + en pause par instance :"
            " envisager --kv-offloading-size (taille à dimensionner sur la RAM hôte) et --kv-offloading-backend native."
        )
        out.warnings.append(Warning(severity="warning", category="kv_limited", rule_id="R-KV-06", message=reason))
    else:
        reason = (
            "Inutile : " + (f"{fmt_int(conc)} contextes P50 tiennent dans le KV" if conc else "capacité du KV inconnue")
            + (f" pour {fmt_int(sessions)} sessions actives + en pause par instance." if sessions else
               " (sessions actives et en pause non fournies).")
        )
    out.decisions.append(Decision(
        key="kv_offloading_size", value=None, rule_id="R-KV-06", title="Offload du KV vers le CPU", justification=reason,
        formula="Seulement si concurrence_KV au contexte typique < sessions actives + sessions en pause (§5)",
        sources=[f"{KB} §5", "vllm/config/cache.py (kv_offloading_size, kv_offloading_backend)"],
        emitted=False,
        omitted_reason=(f"{fmt_int(conc)} contextes P50 tiennent dans le KV, offload inutile" if conc and not out.warnings
                        else "à dimensionner, voir les avertissements" if out.warnings
                        else "capacité du KV inconnue"),
    ))
    return out


def decide_max_model_len(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    out = RuleOutput()
    wl = ctx.request.workload
    max_pos = ctx.model.architecture.max_position_embeddings
    longest = (wl.isl.p99 or wl.isl.p95 or wl.isl.p50) + (wl.osl.p99 or wl.osl.p95 or wl.osl.p50)
    if longest > max_pos:
        out.warnings.append(Warning(
            severity="critical", category="unsupported", rule_id="R-KV-07",
            message=f"ISL + OSL au P99 ({fmt_int(longest)}) dépasse le contexte maximal du modèle ({fmt_int(max_pos)}).",
        ))
    out.decisions.append(Decision(
        key="max_model_len", value=max_pos, rule_id="R-KV-07", title="Longueur de contexte maximale",
        justification=(
            f"Non émis : le défaut est le contexte maximal du modèle ({fmt_int(max_pos)}),"
            f" {'compatible' if longest <= max_pos else 'INCOMPATIBLE'} avec ISL + OSL au P99 ({fmt_int(longest)})."
        ),
        sources=[ctx.model.architecture.source if isinstance(ctx.model.architecture.source, str) else ctx.model.architecture.source[0]],
        emitted=False, omitted_reason=f"défaut = contexte maximal du modèle ({fmt_int(max_pos)})",
    ))
    kv_tokens = state.get("kv_tokens")
    if kv_tokens:
        for label, ctx_len in (("P50", wl.isl.p50), ("P95", wl.isl.p95), ("P99", wl.isl.p99)):
            if ctx_len:
                conc = math.floor(f.kv_concurrency(kv_tokens, ctx_len))
                out.indicators.append(Indicator(
                    key=f"kv_concurrency_{label.lower()}", label=f"Contextes logeables dans le KV à l'ISL {label}",
                    value=conc, formula="KV_tokens / longueur_de_contexte (§3.2)",
                    computation=f"⌊{fmt_int(kv_tokens)} / {fmt_int(ctx_len)}⌋ = {fmt_int(conc)}",
                    confidence="verified" if state.get("kv_source") == "startup_log" else "unverified",
                ))
    return out
