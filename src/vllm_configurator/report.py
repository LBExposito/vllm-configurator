"""Review report: what was asked, what was assumed, what was generated, what is expected and what was measured.

`hypotheses` lists every value the configuration rests on, with where it comes from (entered, read off a trace or a
replay, assumed by the app), so that the person can correct the wrong ones. `build_report` gathers everything a
reviewer (a person or an agent) needs to judge whether the configuration fits the workload, and `render_markdown`
writes it with the questions the reviewer is asked to answer. The texts are in French, like the interface.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

import yaml

from . import render
from .domain import DeploymentRequest, GenerationResult

FORMAT = "vllm-configurator/report"
VERSION = 1

# Where a workload value comes from, as the interface tracks it (field → origin).
ENTERED, TRACE, REPLAY, ASSUMED, CATALOG, LOGS = ("saisie", "trace working-set", "rejeu working-set", "supposée",
                                                 "catalogue", "logs de démarrage")

REVIEW_QUESTIONS = [
    "Les hypothèses sont-elles plausibles pour ce workload et ce matériel ? Signaler en priorité celles d'origine"
    " « supposée » qui pèsent sur le résultat (hit rate, requêtes froides, acceptation, KV, concurrence).",
    "La configuration est-elle cohérente avec ces hypothèses ? Relire chaque paramètre émis et sa justification ;"
    " repérer les contradictions entre paramètres, avec la version de vLLM ou avec le matériel.",
    "Les prédictions (paliers, TTFT, ITL/TPOT, capacité en sessions) sont-elles cohérentes avec les mesures, si elles"
    " existent ? Expliquer les écarts : modèle de latence, forme de la charge, routage, conditions du test.",
    "Les SLO sont-ils atteignables, et le réglage proposé pour plus de sessions est-il crédible ?",
    "Que manque-t-il pour conclure (mesures, données, tests) ?",
]
ANSWER_FORMAT = ("Répondre par : un verdict global (pertinent / à corriger / insuffisant pour conclure), les points"
                 " bloquants, les points à vérifier, et les corrections proposées (paramètre, valeur, raison).")
KNOWN_LIMITS = [
    "L'app ne mesure rien : elle applique des règles. Sans mesure, les latences sont des plages de ×0,8 à ×2,5 issues"
    " d'un modèle roofline dont les rendements sont supposés ; recalées sur une mesure, ±15 % (heuristique).",
    "Le modèle de latence suppose un régime stable et des arrivées indépendantes : ni rafales corrélées (sous-agents,"
    " appels d'outils parallèles), ni boucle fermée près de la saturation, ni dynamique des évictions (seule la capacité"
    " du KV est contrôlée).",
    "Le routage llm-d n'entre dans les latences qu'à travers le hit rate et la part de requêtes froides saisis.",
    "Les poids des scorers llm-d et les coûts du frontal Python sont des heuristiques.",
    "working-set ne mesure pas l'ITL ; son texte rejoué est synthétique, l'acceptation du spec decoding y diffère de la"
    " production.",
]


def _row(label: str, value: Any, origin: str, status: str = "verified", why: str = "", where: str = "") -> dict:
    return {"label": label, "value": value, "origin": origin, "status": status, "why": why, "where": where}


def _pct(p) -> str:
    return " / ".join(f"{v:,}".replace(",", " ") for v in (p.p50, p.p95, p.p99) if v is not None)


def hypotheses(request: DeploymentRequest, result: GenerationResult, origins: dict[str, str] | None = None) -> list[dict]:
    """Every value the configuration and its analysis rest on, with its origin and where to correct it."""
    origins = origins or {}
    wl, spec = request.workload, request.spec
    analysis = result.analysis
    mi = analysis.model_inputs if analysis else {}
    assumed = {a.key: a for a in result.assumptions}
    rows = [
        _row("Type d'usage", wl.usage_type, origins.get("usage_type", ENTERED), where="① Charge"),
        _row("ISL p50 / p95 / p99 (tokens)", _pct(wl.isl), origins.get("isl", ENTERED),
             why=f"ISL moyen retenu : {mi.get('isl_mean_tokens', '—')} tokens." if mi else "", where="① Charge"),
        _row("OSL p50 / p95 / p99 (tokens)", _pct(wl.osl), origins.get("osl", ENTERED),
             why=f"OSL moyen retenu : {mi.get('osl_mean_tokens', '—')} tokens." if mi else "", where="① Charge"),
    ]
    if wl.target_concurrency:
        rows.append(_row("Requêtes en vol visées (toutes instances)", wl.target_concurrency,
                         origins.get("target_concurrency", ENTERED), where="① Charge"))
    elif "target_concurrency" not in assumed:
        rows.append(_row("Requêtes en vol visées", "pleine charge (max-num-seqs par instance)", ASSUMED, "unverified",
                         "Non fournies : l'analyse se place à pleine charge.", "① Charge"))
    if wl.paused_sessions is not None:
        rows.append(_row("Sessions en pause à garder en cache", wl.paused_sessions,
                         origins.get("paused_sessions", ENTERED), where="① Affiner"))
    if wl.session_gap_s is not None:
        rows.append(_row("Attente entre deux requêtes d'une session (s)", wl.session_gap_s,
                         origins.get("session_gap", ENTERED), where="① Affiner"))
    if wl.prefix_cache_hit_rate is not None:
        rows.append(_row("Prefix cache hit rate", wl.prefix_cache_hit_rate, origins.get("hit_rate", ENTERED),
                         where="① Affiner"))
    elif "prefix_cache_hit_rate" in assumed:
        a = assumed["prefix_cache_hit_rate"]
        rows.append(_row("Prefix cache hit rate", a.value, ASSUMED, a.status, a.reason, "① Affiner"))
    cold = mi.get("cold_request_fraction")
    if wl.cold_request_fraction is not None:
        rows.append(_row("Part de requêtes froides", wl.cold_request_fraction, origins.get("cold_fraction", ENTERED),
                         where="① Affiner"))
    elif cold is not None:
        rows.append(_row("Part de requêtes froides", cold, ASSUMED, "unverified",
                         "Heuristique (5 %), plafonnée pour rester compatible avec le hit rate : ce sont elles qui font"
                         " le TTFT p95.", "① Affiner"))
    if wl.warm_uncached_tokens is not None:
        rows.append(_row("Tokens à prefiller d'une requête chaude p50 / p95", _pct(wl.warm_uncached_tokens),
                         origins.get("warm", ENTERED), why=f"σ = {mi.get('warm_uncached_sigma', '—')}.",
                         where="① Affiner"))
    elif mi:
        rows.append(_row("Dispersion des tokens d'une requête chaude (σ)", mi.get("warm_uncached_sigma"), ASSUMED,
                         "unverified", "Heuristique uncached_tokens_sigma.", "① Affiner"))
    shared = mi.get("shared_prefix_tokens", 0)
    rows.append(_row("Préfixe commun à toutes les sessions (tokens)", shared,
                     origins.get("llmd_shared_prefix", ENTERED) if request.routing and request.routing.shared_prefix_tokens
                     is not None else ASSUMED,
                     "verified" if request.routing and request.routing.shared_prefix_tokens is not None else "unverified",
                     "" if shared else "Non fourni : 0, le cas où l'affinité de routage compte le plus.", "① Affiner"))
    if spec.enabled and mi.get("speculative_tokens"):
        if spec.acceptance_length is not None:
            rows.append(_row("Longueur d'acceptation a", spec.acceptance_length,
                             origins.get("acceptance_length", ENTERED), where="④ Mesures"))
        elif "acceptance_length" in assumed:
            a = assumed["acceptance_length"]
            rows.append(_row("Longueur d'acceptation a", a.value, ASSUMED, a.status, a.reason, "④ Mesures"))
    kv = result.facts.kv_tokens_per_instance
    if result.facts.kv_tokens_source == "startup_log":
        rows.append(_row("KV par instance (tokens)", kv, LOGS, where="④ Mesures"))
    elif kv:
        rows.append(_row("KV par instance (tokens)", kv, ASSUMED, "unverified",
                         "Estimation (borne haute) sans logs de démarrage : saisir « GPU KV cache size ».",
                         "④ Mesures"))
    if analysis and analysis.uncertainty:
        calibrated = analysis.calibration is not None
        rows.append(_row(
            "Modèle de latence", "recalé sur la mesure" if calibrated else "non calibré",
            REPLAY if calibrated and origins.get("measurements") == REPLAY else (ENTERED if calibrated else ASSUMED),
            "verified" if calibrated else "unverified",
            f"Incertitude decode ×{analysis.uncertainty[0]:g} à ×{analysis.uncertainty[1]:g}, prefill"
            f" ×{analysis.uncertainty_ttft[0]:g} à ×{analysis.uncertainty_ttft[1]:g}.", "④ Mesures"))
    covered = {"prefix_cache_hit_rate", "acceptance_length"}
    for a in result.assumptions:
        if a.key not in covered:
            rows.append(_row(a.label, a.value, ASSUMED, a.status, a.reason, "① Affiner"))
    unverified = [d.title for d in result.decisions if d.confidence == "unverified" and d.emitted]
    if unverified:
        rows.append(_row("Paramètres fondés sur une donnée à vérifier", ", ".join(unverified), CATALOG, "unverified",
                         "Heuristiques ou comportements non confirmés dans le code source (🔶).", "② Justifications"))
    return rows


def _decision(d) -> dict:
    return {"key": d.key, "title": d.title, "value": d.value, "emitted": d.emitted, "default": d.default,
            "rule": d.rule_id, "confidence": d.confidence, "justification": d.justification, "formula": d.formula,
            "computation": d.computation, "omitted_reason": None if d.emitted else d.omitted_reason,
            "sources": d.sources}


def build_report(result: GenerationResult, catalog, routing=None, epp_yaml: str | None = None,
                 origins: dict[str, str] | None = None, trace_rows: list | None = None, run_profile=None,
                 name: str | None = None) -> dict:
    """Everything a reviewer needs, as a JSON-serializable dict."""
    request = result.request
    analysis = result.analysis
    args = None if result.blocking_errors else yaml.safe_load(render.render_k8s_args(result, catalog))["args"]
    report: dict[str, Any] = {
        "format": FORMAT, "version": VERSION, "name": name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "review": {"questions": REVIEW_QUESTIONS, "answer_format": ANSWER_FORMAT},
        "deployment": {
            "gpu": catalog.gpus[request.gpu_id].display_name, "gpus_per_instance": request.gpus_per_instance,
            "instances": request.num_instances, "model": catalog.models[request.model_id].display_name,
            "model_id": request.model_id, "vllm_version": request.vllm_version,
        },
        "request": request.model_dump(mode="json", exclude_none=True),
        "hypotheses": hypotheses(request, result, origins),
        "configuration": {
            "vllm_args": args, "env": dict(result.env),
            "command": None if result.blocking_errors else render.render_cli(result, catalog),
            "llmd_epp_config": epp_yaml or None, "blocking_errors": list(result.blocking_errors),
        },
        "decisions": [_decision(d) for d in result.decisions],
        "warnings": [{"severity": w.severity, "category": w.category, "message": w.message, "source": w.source}
                     for w in result.warnings],
        "indicators": [{"label": i.label, "value": i.value, "unit": i.unit, "interpretation": i.interpretation}
                       for i in result.indicators],
        "known_limits": KNOWN_LIMITS,
    }
    if routing is not None and not routing.blocking_errors:
        report["llmd"] = {"decisions": [_decision(d) for d in routing.decisions],
                          "warnings": [w.message for w in routing.warnings],
                          "assumptions": [{"label": a.label, "value": a.value, "reason": a.reason}
                                          for a in routing.assumptions]}
    if analysis is not None:
        report["analysis"] = analysis.model_dump(mode="json", exclude={"test_plan"})
        plan = analysis.test_plan
        if plan is not None:
            report["test_plan"] = plan.model_dump(mode="json")
    report["measurements"] = {
        "observed_kv": request.observed_kv.model_dump(exclude_none=True) if request.observed_kv else None,
        "observed_perf": request.observed_perf.model_dump(exclude_none=True) if request.observed_perf else None,
        "replay": None if run_profile is None else {
            "serves": run_profile.serves, "fails_at": run_profile.fails_at, "endpoints": run_profile.endpoints,
            "shared_prefix_tokens": run_profile.shared_prefix_tokens, "notes": run_profile.notes,
            "steps": [{k: v for k, v in asdict(s).items()} for s in run_profile.steps]},
    }
    if trace_rows:
        report["trace"] = [{"label": label, "value": value} for label, value in trace_rows]
    return report


def report_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=1, default=str)


# --- Markdown ---------------------------------------------------------------------------------------------------------

STATUS = {"verified": "✅", "unverified": "🔶"}
VERDICT = {"ok": "✅ OK", "at_risk": "⚠️ à risque", "ko": "🛑 KO", "unknown": "❔ non évaluable"}


def _cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str("—" if value is None or value == "" else value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out + [""]


def _range(pair) -> str:
    if not pair:
        return "—"
    low, high = pair
    fmt = lambda v: "∞" if v == float("inf") else f"{v:g}"  # noqa: E731
    return f"{fmt(low)} – {fmt(high)}"


def render_markdown(report: dict) -> str:
    dep = report["deployment"]
    lines = [
        f"# Rapport de configuration vLLM{' — ' + report['name'] if report.get('name') else ''}",
        "",
        f"Généré le {report['generated_at']} par vllm-configurator. {dep['model']} sur {dep['instances']} instance(s)"
        f" de {dep['gpus_per_instance']} × {dep['gpu']}, vLLM {dep['vllm_version']}.",
        "",
        "## Consignes de relecture",
        "",
        "Ce rapport est destiné à un relecteur (personne ou agent). Il contient la demande, les hypothèses retenues, la"
        " configuration générée et ses justifications, ce que l'app en attend et ce qui a été mesuré. Questions :",
        "",
        *[f"{i}. {q}" for i, q in enumerate(report["review"]["questions"], 1)],
        "",
        report["review"]["answer_format"],
        "",
        "## 1. Hypothèses",
        "",
        "Origine : « saisie » par l'utilisateur, lue sur une « trace working-set » ou un « rejeu working-set », lue"
        " dans les « logs de démarrage », ou « supposée » par l'app (à challenger en priorité).",
        "",
        *_table(["Hypothèse", "Valeur", "Origine", "Statut", "Pourquoi / détail"],
                [[h["label"], h["value"], h["origin"], STATUS.get(h["status"], h["status"]), h["why"]]
                 for h in report["hypotheses"]]),
    ]
    if report.get("trace"):
        lines += ["### Trace working-set importée", "", *_table(["Lu sur la trace", "Valeur"],
                                                                 [[r["label"], r["value"]] for r in report["trace"]])]
    slo = report["request"].get("slo", {})
    lines += ["## 2. Objectifs", "", *_table(["SLO", "Cible"], [
        ["TTFT p95 (ms)", slo.get("ttft_p95_ms")],
        [f"{'ITL' if slo.get('itl_metric', 'itl') == 'itl' else 'TPOT'} p95 (ms)", slo.get("itl_p95_ms")]])]
    conf = report["configuration"]
    lines += ["## 3. Configuration générée", ""]
    if conf["blocking_errors"]:
        lines += [f"- 🛑 {e}" for e in conf["blocking_errors"]] + [""]
    else:
        lines += ["```bash", conf["command"], "```", ""]
        if conf["env"]:
            lines += ["Variables d'environnement : " + ", ".join(f"`{k}={v}`" for k, v in conf["env"].items()), ""]
    lines += ["### Justifications des paramètres", "",
              *_table(["Paramètre", "Valeur", "Émis", "Statut", "Justification", "Calcul"],
                      [[d["title"], d["value"], "oui" if d["emitted"] else "non", STATUS.get(d["confidence"]),
                        d["justification"], d["computation"]] for d in report["decisions"]])]
    if report.get("llmd"):
        lines += ["### Routage llm-d (EndpointPickerConfig)", "", "```yaml", conf["llmd_epp_config"] or "", "```", "",
                  *_table(["Paramètre", "Valeur", "Justification"],
                          [[d["title"], d["value"], d["justification"]] for d in report["llmd"]["decisions"]])]
    warnings = [w for w in report["warnings"] if w["severity"] in ("critical", "warning")]
    if warnings:
        lines += ["### Avertissements", "", *[f"- {'🛑' if w['severity'] == 'critical' else '⚠️'} {w['message']}"
                                              for w in warnings], ""]
    analysis = report.get("analysis")
    if analysis:
        lines += ["## 4. Ce que l'app attend", "", f"Verdict au point de la config : {VERDICT[analysis['verdict']]}."
                  f" {analysis['summary']}", "",
                  *_table(["Contrôle", "Statut", "Cible", "Estimation (plage)", "Mesure", "Détail"],
                          [[c["label"], VERDICT[c["status"]], c["target"],
                            _range((c["estimate_low"], c["estimate_high"])) if c.get("estimate_low") is not None
                            else c["estimate"], c["measured"], c["detail"]] for c in analysis["checks"]])]
        cap = analysis.get("session_capacity")
        if cap:
            lines += ["### Capacité en sessions par instance", "", cap["summary"], "",
                      *_table(["", "Sessions", "Requêtes en vol", "Limite"], [
                          ["Config générée", cap["current_sessions"], cap["current_inflight"], cap["current_binding"]],
                          ["Meilleur réglage", cap["best_sessions"], cap["best_inflight"], cap["best_binding"]],
                          ["Validé par working-set", cap["measured_sessions"], "—", "—"]])]
            if cap["changes"]:
                lines += ["Changements proposés : " + ", ".join(f"`{k} {_cell(v)}`" for k, v in cap["changes"].items()),
                          ""]
        if analysis["bottlenecks"]:
            lines += ["### Goulots", "", *[f"- {VERDICT[b['status']]} **{b['label']}** : {b['explanation']}"
                                           for b in analysis["bottlenecks"]], ""]
        if analysis["recommendations"]:
            lines += ["### Pistes", "", *[f"- **{r['title']}** ({r['kind']}) : {r['rationale']}"
                                          + (f" Changements : {_cell(r['changes'])}" if r["changes"] else "")
                                          for r in analysis["recommendations"]], ""]
        lines += ["### Valeurs sur lesquelles tourne le modèle de latence", "",
                  *_table(["Grandeur", "Valeur"], [[k, v] for k, v in analysis["model_inputs"].items()])]
    plan = report.get("test_plan")
    if plan:
        lines += ["## 5. Plan de test", "", plan["summary"], "",
                  *_table(["Palier", "max-num-seqs", "Requêtes en vol", "Sessions", "ITL p95 (ms)", "TPOT p95 (ms)",
                           "TTFT p95 (ms)", "KV %", "Attendu"],
                          [[("★ " if s["is_generated_config"] else "") + str(s["index"]), s["max_num_seqs"],
                            s["inflight_per_instance"], _range(s["sessions_per_instance"]), _range(s["itl_ms"]),
                            _range(s["tpot_ms"]), _range(s["ttft_ms"]), s["kv_usage_pct"], VERDICT[s["expectation"]]]
                           for s in plan["steps"]]),
                  "Protocole :", "", *[f"{i}. {p}" for i, p in enumerate(plan["protocol"], 1)], ""]
    meas = report["measurements"]
    lines += ["## 6. Mesures", ""]
    if not meas["observed_perf"] and not meas["replay"]:
        lines += ["Aucune mesure de latence fournie : toutes les latences ci-dessus sont des estimations non"
                  " calibrées.", ""]
    if meas["observed_kv"]:
        lines += [f"- Logs de démarrage : {_cell(meas['observed_kv'])}"]
    if meas["observed_perf"]:
        lines += [f"- Latences et charge mesurées : {_cell(meas['observed_perf'])}"]
    if meas["replay"]:
        replay = meas["replay"]
        lines += ["", f"Rejeu working-set : valide {replay['serves']} sessions"
                  + (f", échoue à {replay['fails_at']}" if replay["fails_at"] else "")
                  + f" ({replay['endpoints']} URL, préfixe commun {replay['shared_prefix_tokens']} tokens).", "",
                  *_table(["Sessions", "Validé", "En vol", "TTFT p95 (ms)", "TPOT p95 (ms)", "Hit rate", "Froides",
                           "Évictions", "Raisons"],
                          [[s["sessions"], "oui" if s["passed"] else "non", s["inflight"], s["ttft_p95_ms"],
                            s["tpot_p95_ms"], s["hit_rate"], s["cold_fraction"], s["eviction_share"],
                            "; ".join(s["reasons"])] for s in replay["steps"]]),
                  *[f"- {n}" for n in replay["notes"]]]
    lines += ["", "## 7. Limites connues de l'outil", "", *[f"- {x}" for x in report["known_limits"]], ""]
    if analysis and analysis.get("notes"):
        lines += ["## Annexe : notes de l'analyse", "", *[f"- {n}" for n in analysis["notes"]], ""]
    return "\n".join(lines)
