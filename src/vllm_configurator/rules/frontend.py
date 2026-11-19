"""Frontend rules: --api-server-count and --renderer-num-workers (Python frontend of `vllm serve`)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..domain import Decision, Indicator, Warning
from .context import RuleContext, RuleOutput, fmt_int

RULE_SERVERS, RULE_WORKERS, RULE_CPU = "R-FE-01", "R-FE-02", "R-FE-03"


def _pct(value: float) -> str:
    return "saturé (≥ 100 %)" if value >= 1 else f"{value:.0%}"


def decide_frontend(ctx: RuleContext, state: dict[str, Any]) -> RuleOutput:
    """Size the API server processes and the renderer threads on the traffic of the generated configuration.

    The traffic (requests and streamed outputs per second) comes from the step model of the SLO analysis, at the
    optimistic end of its uncertainty: faster steps mean more traffic for the frontend. vLLM defaults are kept when
    they hold: these flags only cost CPU when the frontend is the bottleneck."""
    from .. import analysis as A  # import local : analysis importe le paquet rules
    from .. import frontend as fe

    out = RuleOutput()
    req = ctx.request
    servers_beh, pool_beh = ctx.behavior("frontend.api_servers"), ctx.behavior("frontend.renderer_pool")
    sources = [servers_beh.source, pool_beh.source]
    caps = A.frontend_caps(ctx, {**state, "api_server_count": None, "renderer_num_workers": None})
    dp = state.get("data_parallel_size") or 1
    default_servers, default_workers = (1 if caps.rust else dp), (ctx.default("renderer_num_workers") or 1)

    def keep(reason: str, confidence: str = "verified") -> RuleOutput:
        for key, title, value in (("api_server_count", "Processus API (frontal)", default_servers),
                                  ("renderer_num_workers", "Threads du renderer", default_workers)):
            out.decisions.append(Decision(
                key=key, value=value, rule_id=RULE_SERVERS if key == "api_server_count" else RULE_WORKERS, title=title,
                justification=reason, emitted=False, omitted_reason=reason, default=value, confidence=confidence,
                sources=sources,
            ))
        out.state.update({"api_server_count": default_servers, "renderer_num_workers": default_workers})
        return out

    if caps.rust:
        out.warnings.append(Warning(
            severity="info", category="assumption", rule_id=RULE_SERVERS,
            message=("Frontend Rust (VLLM_USE_RUST_FRONTEND=1) : vLLM ignore --api-server-count (frontend multithread) et son"
                     " coût CPU n'est pas modélisé ici. Comparer les deux frontends sur un même palier."),
            source=servers_beh.source,
        ))
        return keep("Frontend Rust : --api-server-count ignoré par vLLM, dimensionnement non modélisé.")

    model, _ = A.build_perf_model(ctx, state)
    if model is None:
        return keep("Modèle de latence indisponible : trafic du frontal non estimé, défauts de vLLM conservés.",
                    confidence="unverified")
    env, _ = A.build_env(ctx, state, state.get("kv_tokens"), model)
    env = replace(env, api_servers=default_servers, renderer_workers=default_workers)
    base = A.Point(state["max_num_seqs"], state["max_num_batched_tokens"], state["long_prefill_token_threshold"],
                   req.num_instances)
    fast = A.evaluate(A.with_factor(env, 0), base)
    if fast.frontend is None:
        return keep("Le decode remplit le budget : trafic du frontal non estimé, défauts de vLLM conservés.",
                    confidence="unverified")
    costs, at_default = env.frontend_costs, fast.frontend
    rps, cps = at_default.requests_per_s, at_default.chunks_per_s
    servers, workers = fe.size_frontend(costs, rps, cps, env.isl_mean, caps.max_servers)
    sized = fe.frontend_load(costs, rps, cps, env.isl_mean, env.isl_m2, servers, workers)
    h = ctx.heuristic("frontend_python")
    umax = costs.utilization_max
    shares = ", ".join(f"{name} {share:.0%}" for name, share in at_default.gil_share.items())
    traffic = (f"Trafic par instance (steps rapides de l'estimation, ×{env.spread[0]:g}) : {fmt_int(rps)} requêtes/s de"
               f" {fmt_int(env.isl_mean)} tokens de prompt en moyenne, {fmt_int(cps)} sorties streamées/s"
               f" ({fmt_int(at_default.chunks_per_s / max(rps, 1e-9))} par requête).")

    # --- --api-server-count -------------------------------------------------------
    if servers == default_servers and at_default.gil_utilization > umax:
        reason = (f"{traffic} Avec {default_servers} processus, le GIL est occupé à {_pct(at_default.gil_utilization)}"
                  f" ({shares}), au-delà de {umax:.0%}, mais le nombre de processus API est plafonné"
                  f" ({'; '.join(caps.reasons)}) : le frontal ralentira le TTFT et l'ITL.")
    elif servers == default_servers:
        reason = (f"{traffic} Avec {default_servers} processus, le GIL est occupé à {_pct(at_default.gil_utilization)}"
                  f" ({shares}), sous le seuil de {umax:.0%} : le défaut de vLLM suffit, un processus de plus ne"
                  " ferait que consommer un cœur.")
    else:
        reason = (f"{traffic} Avec {default_servers} processus, le GIL serait occupé à {_pct(at_default.gil_utilization)}"
                  f" ({shares}) : au-delà de {umax:.0%}, les requêtes attendent leur rendu et les sorties leur envoi"
                  f" (TTFT et ITL). {servers} processus ramènent chacun à {_pct(sized.gil_utilization)}. Chaque processus a"
                  " sa boucle asyncio, son GIL, son renderer et sa détokenisation ; l'engine core reste unique.")
    out.decisions.append(Decision(
        key="api_server_count", value=servers, rule_id=RULE_SERVERS, title="Processus API (frontal)",
        justification=reason, formula="⌈charge du GIL / seuil⌉, charge = req/s × (boucle + chat template) + sorties/s × coût d'une sortie",
        computation=f"⌈{at_default.gil_utilization * default_servers:.2f} / {umax:g}⌉ = {servers}" + (
            f" (plafonné : {'; '.join(caps.reasons)})" if caps.reasons else ""),
        confidence=h.status, sources=[*sources, h.source], default=default_servers,
        emitted=servers != default_servers,
        omitted_reason=None if servers != default_servers else "Défaut de vLLM suffisant.",
    ))
    if servers > 1:
        out.warnings.append(Warning(
            severity="info", category="assumption", rule_id=RULE_SERVERS,
            message=(f"{servers} processus API : métriques Prometheus en mode multiprocess (vérifier que l'EPP et les"
                     " tableaux de bord lisent bien vllm:num_requests_waiting et vllm:kv_cache_usage_perc agrégés),"
                     " incompatible avec VLLM_ALLOW_RUNTIME_LORA_UPDATING, et chaque processus charge son tokenizer"
                     " (mémoire hôte)."),
            source=servers_beh.source,
        ))
    if sized.gil_utilization > umax:
        out.warnings.append(Warning(
            severity="critical" if sized.gil_utilization >= 1 else "warning", category="itl", rule_id=RULE_SERVERS,
            message=(f"Frontal limité à {servers} processus API ({'; '.join(caps.reasons) or 'plafond'}) : GIL à"
                     f" {_pct(sized.gil_utilization)}, il ralentira le TTFT et l'ITL. Leviers : plus de CPU par pod, plus"
                     " d'instances, ou comparer le frontend Rust."),
            source=servers_beh.source,
        ))

    # --- --renderer-num-workers ---------------------------------------------------
    pool_default = fe.frontend_load(costs, rps, cps, env.isl_mean, env.isl_m2, servers, default_workers).pool_utilization
    if workers == default_workers:
        reason = (f"Rendu du prompt (chat template + tokenisation) : {fmt_int(costs.render_s(env.isl_mean) * 1e3 - costs.request_loop_s * 1e3)} ms"
                  f" par requête en moyenne ; avec {servers} processus × {default_workers} thread, le pool est occupé à"
                  f" {_pct(pool_default)}, sous {umax:.0%} : le défaut suffit. Le pool ne sert ni les sorties ni le"
                  " multimodal (thread dédié).")
    else:
        reason = (f"Rendu du prompt (chat template + tokenisation) : {fmt_int(costs.render_s(env.isl_mean) * 1e3 - costs.request_loop_s * 1e3)} ms"
                  f" par requête en moyenne, {fmt_int(rps / servers)} requêtes/s par processus : un seul thread serait"
                  f" occupé à {_pct(pool_default)} et les requêtes attendraient leur tokenisation (TTFT). {workers}"
                  f" threads ramènent le pool à {_pct(sized.pool_utilization)}. Gain réel sur la tokenisation, dont le"
                  " cœur Rust tourne hors GIL (tokenizers encode_batch) ; le chat template, en Python, ne se"
                  " parallélise pas par threads.")
    out.decisions.append(Decision(
        key="renderer_num_workers", value=workers, rule_id=RULE_WORKERS, title="Threads du renderer",
        justification=reason, formula="⌈req/s par processus × (chat template + tokenisation) / seuil⌉",
        computation=f"⌈{pool_default * default_workers:.2f} / {umax:g}⌉ = {workers}",
        confidence=h.status, sources=[pool_beh.source, h.source], default=default_workers,
        emitted=workers != default_workers,
        omitted_reason=None if workers != default_workers else "Défaut de vLLM suffisant.",
    ))

    # --- CPU du pod ---------------------------------------------------------------
    tp = req.gpus_per_instance
    hv = h.value
    needed = hv["cores_engine_core"] + tp * hv["cores_per_gpu_worker"] + sized.cores
    out.indicators.append(Indicator(
        key="frontend_cpu", label="Cœurs CPU par instance (engine core, workers GPU, frontal)", value=round(needed, 1),
        unit="cœurs",
        formula="engine core + 1 par worker GPU + frontal (boucle, chat template, tokenisation, sorties)",
        computation=(f"{hv['cores_engine_core']:g} + {tp} × {hv['cores_per_gpu_worker']:g} + {sized.cores:.2f} ="
                     f" {needed:.1f}, plus le système"),
        interpretation=("Fixer la limite CPU du pod au-dessus : une limite trop basse fait brider (CFS) les workers GPU,"
                        " pire qu'un frontal lent." + (f" Limite déclarée : {req.cpu_cores_per_instance:g} cœurs."
                                                        if req.cpu_cores_per_instance else "")),
        confidence=h.status,
    ))
    if req.cpu_cores_per_instance and needed > req.cpu_cores_per_instance:
        out.warnings.append(Warning(
            severity="warning", category="memory", rule_id=RULE_CPU,
            message=(f"CPU du pod : ~{needed:.1f} cœurs nécessaires pour {req.cpu_cores_per_instance:g} disponibles."
                     " Relever la limite CPU avant d'ajouter des processus API."),
        ))
    out.state.update({"api_server_count": servers, "renderer_num_workers": workers})
    return out
