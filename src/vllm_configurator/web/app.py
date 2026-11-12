"""Streamlit UI. Run with: uv run streamlit run src/vllm_configurator/web/app.py"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import yaml
from pydantic import ValidationError

from vllm_configurator import render
from vllm_configurator.catalog import default_catalog
from vllm_configurator.domain import DeploymentRequest, GenerationError
from vllm_configurator.engine import generate

REFERENCE_INPUT = Path(__file__).resolve().parents[3] / "tests" / "golden" / "reference_case" / "input.yaml"
SEVERITY_LABEL = {"critical": "🛑 Critique", "warning": "⚠️ Avertissement", "info": "ℹ️ Info"}
CONFIDENCE_LABEL = {"verified": "✅ vérifié", "unverified": "🔶 à vérifier"}
VERDICT_LABEL = {"ok": "✅ OK", "at_risk": "⚠️ À risque", "ko": "🛑 KO", "unknown": "❔ Non évaluable"}
RECO_LABEL = {"none": "✅", "param": "🎛️ Réglage", "capacity": "🖥️ Capacité GPU", "architecture": "🏗️ Architecture",
              "measure": "📏 Mesure"}
USAGE_ORDER = ["agentic", "chat", "rag", "extraction", "long_summary", "scoring", "batch_offline"]

st.set_page_config(page_title="Générateur de config vLLM", page_icon="⚙️", layout="wide")
catalog = default_catalog()

DEFAULTS = {
    "gpu_id": sorted(catalog.gpus)[0], "gpus_per_instance": 4, "num_instances": 1,
    "model_id": sorted(catalog.models)[0], "vllm_version": catalog.sorted_versions()[-1],
    "usage_type": "agentic", "mode": "interactive", "spec_enabled": True, "draft_sample_method": "défaut vLLM",
    "allow_lossy": False,
}
OPTIONAL = ["isl_p50", "isl_p95", "isl_p99", "osl_p50", "osl_p95", "osl_p99", "hit_rate", "target_concurrency",
            "paused_sessions", "ttft", "itl", "itl_measured", "ttft_measured", "k", "acceptance_length", "kv_tokens", "kv_util", "kv_block", "gpu_util",
            "reserve"]
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
for _key in OPTIONAL:
    st.session_state.setdefault(_key, None)


def load_reference() -> None:
    data = yaml.safe_load(REFERENCE_INPUT.read_text(encoding="utf-8"))
    wl, spec, kv = data["workload"], data.get("spec", {}), data.get("observed_kv", {})
    st.session_state.update({
        "gpu_id": data["gpu_id"], "gpus_per_instance": data["gpus_per_instance"], "num_instances": data["num_instances"],
        "model_id": data["model_id"], "vllm_version": data["vllm_version"],
        "usage_type": wl["usage_type"], "mode": wl["mode"],
        "isl_p50": wl["isl"]["p50"], "isl_p95": wl["isl"].get("p95"), "isl_p99": wl["isl"].get("p99"),
        "osl_p50": wl["osl"]["p50"], "osl_p95": wl["osl"].get("p95"), "osl_p99": wl["osl"].get("p99"),
        "hit_rate": None, "target_concurrency": None, "paused_sessions": None,
        "allow_lossy": data.get("allow_lossy", False), "spec_enabled": True,
        "k": spec.get("num_speculative_tokens"), "draft_sample_method": spec.get("draft_sample_method", "défaut vLLM"),
        "acceptance_length": None, "kv_tokens": kv.get("kv_cache_tokens"), "kv_util": kv.get("gpu_memory_utilization"),
        "kv_block": kv.get("block_size"), "ttft": None, "itl": None, "gpu_util": None, "reserve": None,
        "itl_measured": None, "ttft_measured": None,
    })


def opt_int(label: str, key: str, help: str | None = None, min_value: int = 1) -> int | None:
    value = st.number_input(label, min_value=min_value, value=None, step=1, key=key, help=help)
    return int(value) if value is not None else None


def opt_float(label: str, key: str, min_value: float, max_value: float | None = None, step: float = 0.01,
              help: str | None = None) -> float | None:
    value = st.number_input(label, min_value=float(min_value), max_value=None if max_value is None else float(max_value),
                            value=None, step=step, key=key, help=help, format="%g")
    return float(value) if value is not None else None


# --- Sidebar: inputs -------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Config vLLM")
    st.button("Charger le cas de référence (§1)", on_click=load_reference, use_container_width=True)

    st.header("Matériel")
    gpu_id = st.selectbox("GPU", sorted(catalog.gpus), key="gpu_id", format_func=lambda g: catalog.gpus[g].display_name)
    c1, c2 = st.columns(2)
    gpus_per_instance = c1.number_input("GPU par instance (TP)", min_value=1, max_value=64, key="gpus_per_instance")
    num_instances = c2.number_input("Instances", min_value=1, max_value=1024, key="num_instances")

    st.header("Modèle et version")
    model_id = st.selectbox("Modèle", sorted(catalog.models), key="model_id", format_func=lambda m: catalog.models[m].display_name)
    versions = catalog.sorted_versions()
    vllm_version = st.selectbox("Version de vLLM", versions, key="vllm_version")

    st.header("Profil de charge")
    usage_type = st.selectbox("Type d'usage", USAGE_ORDER, key="usage_type",
                              format_func=lambda u: catalog.workload.profiles[u].label)
    mode = st.radio("Mode", ["interactive", "batch"], horizontal=True, key="mode",
                    format_func=lambda m: {"interactive": "Interactif", "batch": "Batch"}[m])
    st.caption("ISL (tokens d'entrée par requête)")
    c1, c2, c3 = st.columns(3)
    with c1:
        isl_p50 = opt_int("P50", "isl_p50")
    with c2:
        isl_p95 = opt_int("P95", "isl_p95")
    with c3:
        isl_p99 = opt_int("P99", "isl_p99")
    st.caption("OSL (tokens générés par requête, raisonnement compris)")
    c1, c2, c3 = st.columns(3)
    with c1:
        osl_p50 = opt_int("P50 ", "osl_p50")
    with c2:
        osl_p95 = opt_int("P95 ", "osl_p95")
    with c3:
        osl_p99 = opt_int("P99 ", "osl_p99")
    hit_rate = opt_float("Prefix cache hit attendu (0–1)", "hit_rate", 0.0, 1.0,
                         help="Vide : valeur publiée pour le type d'usage, affichée comme hypothèse.")
    target_concurrency = opt_int("Concurrence cible (toutes instances)", "target_concurrency")
    paused_sessions = opt_int("Sessions en pause à garder en cache", "paused_sessions", min_value=0)

    st.header("SLO")
    c1, c2 = st.columns(2)
    with c1:
        ttft = opt_float("TTFT p95 (ms)", "ttft", 1.0, step=10.0)
    with c2:
        itl = opt_float("ITL p95 (ms)", "itl", 1.0, step=1.0)

    st.caption("Les SLO alimentent l'onglet « Interprétation SLO » : verdict, goulot et réglage correctif.")

    st.header("Speculative decoding")
    spec_enabled = st.checkbox("Activer la méthode native du modèle", key="spec_enabled")
    k = opt_int("num_speculative_tokens (k)", "k", help="Vide : valeur recommandée pour le modèle.")
    draft_choice = st.selectbox("draft_sample_method", ["défaut vLLM", "greedy", "probabilistic"], key="draft_sample_method")
    acceptance = opt_float("Longueur d'acceptation mesurée a", "acceptance_length", 1.0, step=0.1,
                           help="a = 1 + spec_decode_num_accepted_tokens / spec_decode_num_drafts")
    allow_lossy = st.checkbox("Tolérer une vérification lossy (FLy)", key="allow_lossy")

    st.header("Logs de démarrage (optionnel)")
    kv_tokens = opt_int("GPU KV cache size (tokens)", "kv_tokens", help="Ligne « GPU KV cache size: N tokens » d'une instance.")
    kv_util = opt_float("gpu-memory-utilization lors de la mesure", "kv_util", 0.05, 1.0)
    kv_block = opt_int("kv cache block size (log)", "kv_block")

    st.header("Mesures en production (optionnel)")
    st.caption("Mesurées avec la config générée : elles priment sur l'estimation, et l'ITL recalibre le modèle de latence.")
    c1, c2 = st.columns(2)
    with c1:
        itl_measured = opt_float("ITL p95 mesurée (ms)", "itl_measured", 0.1, step=1.0)
    with c2:
        ttft_measured = opt_float("TTFT p95 mesuré (ms)", "ttft_measured", 1.0, step=10.0)

    with st.expander("Avancé"):
        gpu_util = opt_float("Forcer gpu-memory-utilization", "gpu_util", 0.05, 1.0)
        reserve = opt_float("Réserve par GPU pour l'estimation du KV (Gio)", "reserve", 0.0, step=1.0,
                            help="Activations, CUDA graphs, drafter. Vide : 0, l'estimation est alors une borne haute.")


# --- Generation ------------------------------------------------------------

st.title("Générateur de configuration vLLM")

missing = [label for label, value in (("ISL P50", isl_p50), ("OSL P50", osl_p50)) if value is None]
if missing:
    st.info("Renseignez au minimum " + " et ".join(missing) + " (ou chargez le cas de référence dans la barre latérale).")
    st.stop()


def pct(p50, p95, p99) -> dict:
    return {key: value for key, value in (("p50", p50), ("p95", p95), ("p99", p99)) if value is not None}


try:
    request = DeploymentRequest(
        gpu_id=gpu_id, gpus_per_instance=int(gpus_per_instance), num_instances=int(num_instances), model_id=model_id,
        vllm_version=vllm_version,
        workload={"usage_type": usage_type, "mode": mode, "isl": pct(isl_p50, isl_p95, isl_p99),
                  "osl": pct(osl_p50, osl_p95, osl_p99), "prefix_cache_hit_rate": hit_rate,
                  "target_concurrency": target_concurrency, "paused_sessions": paused_sessions},
        slo={"ttft_p95_ms": ttft, "itl_p95_ms": itl}, allow_lossy=allow_lossy,
        spec={"enabled": spec_enabled, "num_speculative_tokens": k, "acceptance_length": acceptance,
              "draft_sample_method": None if draft_choice == "défaut vLLM" else draft_choice},
        observed_kv=({"kv_cache_tokens": kv_tokens, "gpu_memory_utilization": kv_util, "block_size": kv_block}
                     if kv_tokens else None),
        observed_perf=({"itl_p95_ms": itl_measured, "ttft_p95_ms": ttft_measured}
                       if itl_measured or ttft_measured else None),
        gpu_memory_utilization=gpu_util, kv_estimate_reserve_gib_per_gpu=reserve,
    )
    result = generate(request, catalog)
except ValidationError as exc:
    st.error("Entrées invalides :\n\n" + "\n".join(f"- {'.'.join(map(str, e['loc']))} : {e['msg']}" for e in exc.errors()))
    st.stop()
except GenerationError as exc:
    st.error("\n\n".join(exc.errors))
    st.stop()

if result.blocking_errors:
    for error in result.blocking_errors:
        st.error(error)
    st.stop()

critical = [w for w in result.warnings if w.severity == "critical"]
to_verify = [w for w in result.warnings if w.category == "to_verify"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("max-num-seqs", result.facts.max_num_seqs)
c2.metric("max-num-batched-tokens", result.facts.max_num_batched_tokens)
c3.metric("KV par instance", f"{result.facts.kv_tokens_per_instance:,}".replace(",", " ") if result.facts.kv_tokens_per_instance else "—",
          help={"startup_log": "Lu dans les logs", "estimate": "Estimation (à vérifier)"}.get(result.facts.kv_tokens_source))
c4.metric("Avertissements", f"{len(critical)} critiques · {len(to_verify)} à vérifier")
for w in critical:
    st.error(w.message)

analysis = result.analysis
if analysis is not None:
    banner = {"ok": st.success, "at_risk": st.warning, "ko": st.error, "unknown": st.info}[analysis.verdict]
    banner(f"**Interprétation SLO — {VERDICT_LABEL[analysis.verdict]}** : {analysis.summary}"
           " Détail et réglage correctif dans l'onglet « Interprétation SLO ».")

tabs = st.tabs(["Configuration", "Interprétation SLO", "Justifications", "Avertissements", "Indicateurs", "Hypothèses",
                "Vérification et métriques"])

with tabs[0]:
    k8s, cli = render.render_k8s_args(result, catalog), render.render_cli(result, catalog)
    st.subheader("YAML `args` (KServe / Kubernetes)")
    st.code(k8s, language="yaml")
    st.download_button("Télécharger args.yaml", k8s, file_name="vllm-args.yaml")
    st.subheader("Ligne de commande")
    st.code(cli, language="bash")

with tabs[1]:
    if analysis is None:
        st.info("Analyse indisponible.")
    else:
        st.subheader(f"Verdict : {VERDICT_LABEL[analysis.verdict]}")
        st.write(analysis.summary)
        st.dataframe([{
            "Contrôle": c.label, "Statut": VERDICT_LABEL[c.status],
            "Cible": "—" if c.target is None else f"{c.target:g} {c.unit or ''}".strip(),
            "Estimation": "—" if c.estimate is None else ("∞" if c.estimate == float("inf") else f"{c.estimate:g} {c.unit or ''}".strip()),
            "Mesure": "—" if c.measured is None else f"{c.measured:g} {c.unit or ''}".strip(),
            "Verdict": "oui" if c.in_verdict else "non (cas pire)", "Détail": c.detail,
        } for c in analysis.checks], use_container_width=True, hide_index=True)

        st.subheader("Goulots d'étranglement")
        if not analysis.bottlenecks:
            st.success("Aucun goulot identifié au point de fonctionnement.")
        for b in analysis.bottlenecks:
            (st.error if b.status == "ko" else st.warning)(f"**{b.label}** — {b.explanation}")

        st.subheader("Recommandations")
        for i, r in enumerate(analysis.recommendations, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {RECO_LABEL[r.kind]} — {r.title}**")
                st.write(r.rationale)
                if r.changes:
                    st.code(chr(10).join(
                        f"{key}: {json.dumps(value, separators=(',', ':')) if isinstance(value, dict) else value}"
                        for key, value in r.changes.items()), language="yaml")

        if analysis.step_breakdown:
            with st.expander("Décomposition du step au point de fonctionnement"):
                st.dataframe([{"Grandeur": k, "Valeur": str(v)} for k, v in analysis.step_breakdown.items()],
                             use_container_width=True, hide_index=True)
        st.caption(f"{CONFIDENCE_LABEL[analysis.confidence]} : modèle d'ordre de grandeur, à confirmer par la mesure.")
        for note in analysis.notes:
            st.caption(note)

with tabs[2]:
    for d in result.decisions:
        value = json.dumps(d.value, ensure_ascii=False) if isinstance(d.value, dict) else d.value
        status = "émis" if d.emitted else "non émis"
        with st.expander(f"{d.title} : {value if value is not None else '—'}  ·  {status}  ·  {CONFIDENCE_LABEL[d.confidence]}"):
            st.markdown(f"**Règle** `{d.rule_id}` · **paramètre** `{d.key}`"
                        + (f" · **défaut vLLM** `{d.default}`" if d.default is not None else ""))
            st.write(d.justification)
            if d.formula:
                st.markdown(f"**Formule** : {d.formula}")
            if d.computation:
                st.markdown(f"**Calcul** : `{d.computation}`")
            if not d.emitted and d.omitted_reason:
                st.markdown(f"**Non émis** : {d.omitted_reason}")
            if d.sources:
                st.caption("Sources : " + " · ".join(d.sources))

with tabs[3]:
    if not result.warnings:
        st.success("Aucun avertissement.")
    for severity in ("critical", "warning", "info"):
        group = [w for w in result.warnings if w.severity == severity]
        if not group:
            continue
        st.subheader(f"{SEVERITY_LABEL[severity]} ({len(group)})")
        for w in group:
            prefix = "🔶 " if w.category == "to_verify" else ""
            box = {"critical": st.error, "warning": st.warning, "info": st.info}[severity]
            box(prefix + w.message + (f"\n\n*Source : {w.source}*" if w.source else ""))

with tabs[4]:
    rows = [{"Indicateur": i.label, "Valeur": i.value, "Unité": i.unit or "", "Formule": i.formula or "",
             "Calcul": i.computation or "", "Lecture": i.interpretation or "", "Statut": CONFIDENCE_LABEL[i.confidence]}
            for i in result.indicators]
    st.dataframe(rows, use_container_width=True, hide_index=True)

with tabs[5]:
    if not result.assumptions:
        st.success("Aucune hypothèse : toutes les entrées utiles sont fournies.")
    for a in result.assumptions:
        st.warning(f"**{a.label}** = `{a.value}` ({CONFIDENCE_LABEL[a.status]})\n\n{a.reason}")

with tabs[6]:
    st.subheader("Vérifier les flags sur l'image déployée")
    st.caption("Le frontend Rust de vLLM ≥ 0.33 échoue sur un flag inconnu : vérifier chaque flag sur l'image réelle.")
    st.code(render.render_verify_command(result, catalog), language="bash")
    st.subheader("Métriques à surveiller")
    st.dataframe([{"Métrique": m.name, "Usage": m.usage} for m in catalog.workload.metrics],
                 use_container_width=True, hide_index=True)
    for test in catalog.workload.validation_tests:
        st.info(test)
    with st.expander("Faits transmis aux étapes suivantes (v2 : EPP llm-d)"):
        st.json(result.facts.model_dump())
