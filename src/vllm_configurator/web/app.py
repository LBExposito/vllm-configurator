"""Streamlit UI, as a five-step workflow. Run with: uv run streamlit run src/vllm_configurator/web/app.py

① describe the need → ② the app's assumptions and the proposed config → ③ test it → ④ import the measurements, the
app recalibrates and proposes a better setting → ⑤ export the report for a reviewer.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Always run the repository's package, not a copy installed in the venv: a host (Streamlit Community Cloud) installs the
# project on the first deployment, then updates the repository without reinstalling it. If the code or the catalog
# changed since the last import in this process, the modules are purged to be read again.
_SRC = Path(__file__).resolve().parents[2]
_ROOT = _SRC.parent
sys.path[:] = [str(_SRC)] + [entry for entry in sys.path if Path(entry or ".").resolve() != _SRC]


def _source_stamp() -> float:
    files = list((_SRC / "vllm_configurator").rglob("*.py")) + list((_ROOT / "catalog").rglob("*.yaml"))
    return max((f.stat().st_mtime for f in files), default=0.0)


def _refresh_package() -> None:
    stamp = _source_stamp()
    loaded = sys.modules.get("vllm_configurator")
    stale_copy = loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(_SRC)
    if stale_copy or getattr(sys, "_vllm_configurator_stamp", None) != stamp:
        for name in [m for m in sys.modules if m == "vllm_configurator" or m.startswith("vllm_configurator.")]:
            del sys.modules[name]
        sys._vllm_configurator_stamp = stamp


_refresh_package()

import streamlit as st  # noqa: E402
import yaml  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from vllm_configurator import render, report, traces  # noqa: E402
from vllm_configurator.catalog import default_catalog  # noqa: E402
from vllm_configurator.domain import DeploymentRequest, GenerationError  # noqa: E402
from vllm_configurator.engine import generate  # noqa: E402
from vllm_configurator.llmd import generate_routing, render_epp_config, render_vllm_args  # noqa: E402
from vllm_configurator.scenario import (  # noqa: E402
    ROUTING_WEIGHT_KEYS, ScenarioError, dump_scenario, load_scenario, request_to_form, run_to_form, trace_to_form,
)

REFERENCE_INPUT = Path(__file__).resolve().parents[3] / "tests" / "golden" / "reference_case" / "input.yaml"
STEPS = ["① Votre besoin", "② Hypothèses et config", "③ Tester", "④ Mesurer et ajuster", "⑤ Exporter"]
SEVERITY_LABEL = {"critical": "🛑 Critique", "warning": "⚠️ Avertissement", "info": "ℹ️ Info"}
CONFIDENCE_LABEL = {"verified": "✅ vérifié", "unverified": "🔶 à vérifier"}
VERDICT_LABEL = {"ok": "✅ OK", "at_risk": "⚠️ À risque", "ko": "🛑 KO", "unknown": "❔ Non évaluable"}
RECO_LABEL = {"none": "✅", "param": "🎛️ Réglage", "capacity": "🖥️ Capacité GPU", "architecture": "🏗️ Architecture",
              "measure": "📏 Mesure"}
USAGE_ORDER = ["agentic", "chat", "rag", "extraction", "long_summary", "scoring", "batch_offline"]

st.set_page_config(page_title="Configurateur vLLM", page_icon="⚙️", layout="wide")
catalog = default_catalog()

DEFAULTS = {
    "gpu_id": sorted(catalog.gpus)[0], "gpus_per_instance": 4, "num_instances": 1,
    "model_id": sorted(catalog.models)[0], "vllm_version": catalog.sorted_versions()[-1],
    "usage_type": "agentic", "mode": "interactive", "spec_enabled": True, "draft_sample_method": "défaut vLLM",
    "allow_lossy": False, "itl_metric": "itl",
    "llmd_version": sorted(catalog.llmd_versions)[-1], "llmd_mode": "approximate", "llmd_kv_endpoint": "",
    "scenario_name": "", "extra_args_text": "", "extra_env_text": "", "llmd_params_text": "",
    "trace_reply_cached": True, "step": STEPS[0],
}
OPTIONAL = ["isl_p50", "isl_p95", "isl_p99", "osl_p50", "osl_p95", "osl_p99", "hit_rate", "cold_fraction",
            "target_concurrency", "paused_sessions", "warm_p50", "warm_p95", "session_gap", "trace_sessions",
            "ttft", "itl", "itl_measured", "ttft_measured", "tpot_measured", "inflight_measured", "sessions_measured",
            "capacity_measured", "k", "acceptance_length", "kv_tokens", "kv_util", "kv_block", "gpu_util", "reserve",
            "cpu_cores", "max_model_len", "llmd_shared_prefix", "llmd_obs_block", "llmd_obs_lru", "llmd_obs_window",
            "llmd_obs_hit", "run_step", *ROUTING_WEIGHT_KEYS.values()]
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
for _key in OPTIONAL:
    st.session_state.setdefault(_key, None)
st.session_state.setdefault("origins", {})
# Streamlit drops the state of a widget that is not rendered in a run: re-assigning it keeps the values of the steps
# that are not shown.
for _key in [*DEFAULTS, *OPTIONAL]:
    st.session_state[_key] = st.session_state[_key]


# --- Callbacks ---------------------------------------------------------------------------------------------------------

def go(index: int) -> None:
    st.session_state["step"] = STEPS[index]


def mark(origin_key: str) -> None:
    st.session_state["origins"][origin_key] = report.ENTERED


def apply_request(request: DeploymentRequest) -> list[str]:
    """Fill the form from a request. Returns the problems that prevent it (unknown catalog entries)."""
    problems = []
    for label, value, known in (("GPU", request.gpu_id, catalog.gpus), ("Modèle", request.model_id, catalog.models),
                                ("Version de vLLM", request.vllm_version, catalog.vllm_versions)):
        if value not in known:
            problems.append(f"{label} {value!r} absent du catalogue (disponibles : {', '.join(sorted(known))}).")
    if request.routing and request.routing.llmd_version and request.routing.llmd_version not in catalog.llmd_versions:
        problems.append(f"llm-d {request.routing.llmd_version!r} absent du catalogue"
                        f" (disponibles : {', '.join(sorted(catalog.llmd_versions))}).")
    if not problems:
        form = request_to_form(request)
        form["llmd_version"] = form["llmd_version"] or DEFAULTS["llmd_version"]
        st.session_state.update(form)
        st.session_state["origins"] = {}
    return problems


def load_reference() -> None:
    request, _ = load_scenario(REFERENCE_INPUT.read_text(encoding="utf-8"))
    apply_request(request)
    st.session_state["scenario_name"] = "Cas de référence (§1)"
    go(1)


def import_scenario() -> None:
    uploaded = st.session_state.get("scenario_file")
    if uploaded is None:
        return
    try:
        request, name = load_scenario(uploaded.getvalue())
    except ScenarioError as exc:
        st.session_state["scenario_message"] = ("error", f"{uploaded.name} : {exc}")
        return
    problems = apply_request(request)
    if problems:
        st.session_state["scenario_message"] = ("error", f"{uploaded.name} : " + " ".join(problems))
        return
    st.session_state["scenario_name"] = name or Path(uploaded.name).stem
    st.session_state["scenario_message"] = ("success", f"Scénario « {st.session_state['scenario_name']} » chargé.")


TRACE_ORIGINS = {"isl_p50": "isl", "osl_p50": "osl", "hit_rate": "hit_rate", "cold_fraction": "cold_fraction",
                 "warm_p50": "warm", "session_gap": "session_gap", "target_concurrency": "target_concurrency",
                 "paused_sessions": "paused_sessions"}


def import_trace() -> None:
    """Fill the workload from a working-set trace (sizes, cache, waits), with the current shared prefix."""
    uploaded = st.session_state.get("trace_file")
    if uploaded is None:
        return
    try:
        profile = traces.profile_trace(uploaded.getvalue(), st.session_state.get("llmd_shared_prefix") or 0,
                                       bool(st.session_state.get("trace_reply_cached", True)))
    except traces.TraceError as exc:
        st.session_state["trace_message"] = ("error", f"{uploaded.name} : {exc}")
        return
    form = trace_to_form(profile, st.session_state.get("trace_sessions"))
    st.session_state.update(form)
    st.session_state["origins"].update({TRACE_ORIGINS[k]: report.TRACE for k in form if k in TRACE_ORIGINS})
    st.session_state["trace_summary"] = (traces.summary_rows(profile), profile.notes)
    st.session_state["trace_message"] = ("success", f"Trace « {uploaded.name} » appliquée à la charge.")


def run_instances(profile) -> int:
    """vLLM instances the replay measured: one per URL, else the instances of the form (a gateway in front)."""
    return profile.endpoints if profile.endpoints > 1 else int(st.session_state.get("num_instances") or 1)


def apply_run() -> None:
    """Fill the measurements from the selected step of the imported replay."""
    profile = st.session_state.get("run_profile")
    if profile is None:
        return
    index = st.session_state.get("run_step")
    index = profile.default_step() if index is None or index >= len(profile.steps) else index
    form = run_to_form(profile, index, run_instances(profile), st.session_state.get("itl_metric", "itl"))
    st.session_state.update(form)
    origins = st.session_state["origins"]
    origins["measurements"] = report.REPLAY
    for key in ("hit_rate", "cold_fraction"):
        if form.get(key) is not None:
            origins[key] = report.REPLAY
    st.session_state["run_message"] = ("success", f"Palier de {profile.steps[index].sessions} sessions appliqué aux"
                                                  " mesures : le modèle est recalé dessus.")


def import_run() -> None:
    uploaded = st.session_state.get("run_file")
    if uploaded is None:
        return
    try:
        profile = traces.profile_run(uploaded.getvalue())
    except traces.TraceError as exc:
        st.session_state["run_message"] = ("error", f"{uploaded.name} : {exc}")
        return
    st.session_state["run_profile"] = profile
    st.session_state["run_step"] = profile.default_step()
    apply_run()


# --- Widgets -----------------------------------------------------------------------------------------------------------

def opt_int(label: str, key: str, help: str | None = None, min_value: int = 1, origin: str | None = None) -> None:
    st.number_input(label, min_value=min_value, value=None, step=1, key=key, help=help, on_change=mark,
                    args=(origin or key,))


def opt_float(label: str, key: str, min_value: float, max_value: float | None = None, step: float = 0.01,
              help: str | None = None, origin: str | None = None) -> None:
    st.number_input(label, min_value=float(min_value), max_value=None if max_value is None else float(max_value),
                    value=None, step=step, key=key, help=help, format="%g", on_change=mark, args=(origin or key,))


def message(key: str) -> None:
    msg = st.session_state.pop(key, None)
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])


def next_button(index: int, label: str) -> None:
    st.button(f"{label} →", on_click=go, args=(index,), type="primary", key=f"next_{index}")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "scenario"


def fmt_ms(value: float) -> str:
    if value == float("inf"):
        return "∞"
    return f"{value / 1e3:.1f} s" if value >= 1000 else f"{value:.0f} ms"


def fmt_range(pair, unit_ms: bool = True) -> str:
    if pair is None:
        return "—"
    low, high = pair
    if unit_ms:
        return f"{fmt_ms(low)} – {fmt_ms(high)}"
    return f"{low:,.0f} – {high:,.0f}".replace(",", " ")


def first_sentence(text: str, limit: int = 220) -> str:
    head = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return head if len(head) <= limit else head[:limit].rsplit(" ", 1)[0] + "…"


def show_decision(d, default_label: str) -> None:
    value = json.dumps(d.value, ensure_ascii=False) if isinstance(d.value, dict) else d.value
    status = "émis" if d.emitted else "non émis"
    with st.expander(f"{d.title} : {value if value is not None else '—'}  ·  {status}  ·  {CONFIDENCE_LABEL[d.confidence]}"):
        st.markdown(f"**Règle** `{d.rule_id}` · **paramètre** `{d.key}`"
                    + (f" · **{default_label}** `{d.default}`" if d.default is not None else ""))
        st.write(d.justification)
        if d.formula:
            st.markdown(f"**Formule** : {d.formula}")
        if d.computation:
            st.markdown(f"**Calcul** : `{d.computation}`")
        if not d.emitted and d.omitted_reason:
            st.markdown(f"**Non émis** : {d.omitted_reason}")
        if d.sources:
            st.caption("Sources : " + " · ".join(d.sources))


def show_warnings(warnings) -> None:
    for severity in ("critical", "warning", "info"):
        for w in [w for w in warnings if w.severity == severity]:
            box = {"critical": st.error, "warning": st.warning, "info": st.info}[severity]
            box(("🔶 " if w.category == "to_verify" else "") + w.message + (f"\n\n*Source : {w.source}*" if w.source else ""))


def checks_table(analysis, with_measure: bool) -> None:
    st.dataframe([{
        "Contrôle": c.label, "Statut": VERDICT_LABEL[c.status],
        "Cible": "—" if c.target is None else f"{c.target:g} {c.unit or ''}".strip(),
        "Estimation": ("—" if c.estimate is None else
                       fmt_range((c.estimate_low, c.estimate_high)) if c.estimate_low is not None and c.unit == "ms"
                       else f"{c.estimate:g} {c.unit or ''}".strip()),
        **({"Mesure": "—" if c.measured is None else f"{c.measured:g} {c.unit or ''}".strip()} if with_measure else {}),
        "Détail": first_sentence(c.detail, 260),
    } for c in analysis.checks if c.in_verdict], width="stretch", hide_index=True)


def capacity_block(capacity) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Config générée (estimée)", capacity.current_sessions,
              help=f"{capacity.current_inflight} requêtes en vol ; limite : {capacity.current_binding}.")
    c2.metric("Meilleur réglage vLLM (estimé)", capacity.best_sessions,
              delta=capacity.best_sessions - capacity.current_sessions or None,
              help=f"{capacity.best_inflight} requêtes en vol ; limite : {capacity.best_binding}.")
    c3.metric("Validé par working-set", "—" if capacity.measured_sessions is None else f"{capacity.measured_sessions:g}")
    st.write(capacity.summary)
    if capacity.changes:
        st.markdown("**Flags à changer pour le meilleur réglage**")
        st.code(yaml.safe_dump(capacity.changes, allow_unicode=True, sort_keys=False, default_flow_style=None),
                language="yaml")
    for note in capacity.notes:
        st.caption(note)


# --- Header ------------------------------------------------------------------------------------------------------------

st.title("⚙️ Configurateur vLLM")
st.caption("Décrivez votre besoin, l'app fait ses hypothèses et propose une config ; vous la testez, vous importez les"
           " mesures, elle recale son modèle et propose un meilleur réglage. Exportez le rapport pour le faire relire.")
st.radio("Étape", STEPS, key="step", horizontal=True, label_visibility="collapsed")
step = STEPS.index(st.session_state["step"])
st.divider()


# --- ① Votre besoin ----------------------------------------------------------------------------------------------------

if step == 0:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.button("Charger le cas de référence (§1)", on_click=load_reference, key="load_reference",
                  help="DeepSeek V4.1 Flash sur 2 × 4 B300, charge agentique : pour découvrir l'app.")
    with c2:
        with st.expander("💾 Reprendre un scénario enregistré (YAML)"):
            st.file_uploader("Scénario", type=["yaml", "yml"], key="scenario_file", on_change=import_scenario,
                             help="Fichier exporté à l'étape ⑤, ou requête nue (tests/golden/reference_case/input.yaml).")
    message("scenario_message")

    st.subheader("Matériel et modèle")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 2, 1])
    c1.selectbox("GPU", sorted(catalog.gpus), key="gpu_id", format_func=lambda g: catalog.gpus[g].display_name)
    c2.number_input("GPU par instance (TP)", min_value=1, max_value=64, key="gpus_per_instance")
    c3.number_input("Instances", min_value=1, max_value=1024, key="num_instances")
    c4.selectbox("Modèle", sorted(catalog.models), key="model_id", format_func=lambda m: catalog.models[m].display_name)
    c5.selectbox("Version de vLLM", catalog.sorted_versions(), key="vllm_version")

    st.subheader("Charge")
    st.selectbox("Type d'usage", USAGE_ORDER, key="usage_type", format_func=lambda u: catalog.workload.profiles[u].label,
                 on_change=mark, args=("usage_type",))
    with st.expander("📼 Importer une trace working-set (ws trace) — recommandé pour des sessions agentiques",
                     expanded=st.session_state["usage_type"] == "agentic" and not st.session_state.get("trace_summary")):
        st.caption("Une trace `ws trace pi|claude` (tailles et attentes seulement) remplit les longueurs, le cache, les"
                   " requêtes froides et l'attente entre requêtes. Le préfixe commun utilisé est celui d'« Affiner ».")
        c1, c2 = st.columns(2)
        with c1:
            opt_int("Sessions visées (toutes instances)", "trace_sessions",
                    help="Avec les durées de service de la trace : requêtes en vol et sessions en pause.")
        with c2:
            st.checkbox("La réponse précédente reste en cache", key="trace_reply_cached",
                        help="Décocher si le chat template retire la réponse précédente (raisonnement) de l'historique.")
        st.file_uploader("Trace (.json)", type=["json"], key="trace_file", on_change=import_trace)
        st.button("Réappliquer la trace", on_click=import_trace, key="reapply_trace")
        message("trace_message")
        summary = st.session_state.get("trace_summary")
        if summary:
            st.dataframe([{"Lu sur la trace": label, "Valeur": value} for label, value in summary[0]],
                         hide_index=True, width="stretch")
            for note in summary[1]:
                st.caption(note)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        opt_int("ISL P50 (tokens)", "isl_p50", help="Tokens d'entrée par appel au modèle (pas par session).", origin="isl")
    with c2:
        opt_int("ISL P95", "isl_p95", origin="isl")
    with c3:
        opt_int("OSL P50 (tokens)", "osl_p50", help="Tokens générés par appel, raisonnement compris.", origin="osl")
    with c4:
        opt_int("OSL P95", "osl_p95", origin="osl")
    with c5:
        opt_int("Requêtes en vol visées", "target_concurrency",
                help=("Toutes instances, running + waiting : pas des utilisateurs (un agent n'a une requête en cours"
                      " qu'une partie du temps). Vide : pleine charge."))

    st.subheader("Objectifs (SLO)")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        opt_float("TTFT p95 (ms)", "ttft", 1.0, step=10.0)
    with c2:
        opt_float("ITL / TPOT p95 (ms)", "itl", 1.0, step=1.0)
    with c3:
        st.radio("La cible de génération porte sur", ["itl", "tpot"], horizontal=True, key="itl_metric",
                 format_func=lambda m: {"itl": "ITL (intervalle entre sorties)", "tpot": "TPOT (temps par token)"}[m],
                 help=("ITL = vllm:inter_token_latency_seconds (un step) ; TPOT ="
                       " vllm:request_time_per_output_token_seconds, divisé par ~a avec le spec decoding. Pour le"
                       " ressenti utilisateur, c'est le TPOT qui compte."))

    with st.expander("Affiner (facultatif) : cache, sessions, spec decoding, contexte, routage, CPU"):
        st.markdown("**Charge**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            opt_int("ISL P99", "isl_p99", origin="isl")
            opt_int("OSL P99", "osl_p99", origin="osl")
        with c2:
            opt_float("Prefix cache hit attendu (0–1)", "hit_rate", 0.0, 1.0,
                      help="Vide : valeur publiée pour le type d'usage (hypothèse affichée à l'étape ②).")
            opt_float("Part de requêtes froides (0–1)", "cold_fraction", 0.0, 1.0,
                      help="Préfixe absent du cache (nouvelle session, éviction, mauvais routage). Vide : 5 % plafonnés.")
        with c3:
            opt_int("Tokens d'une requête chaude P50", "warm_p50", origin="warm",
                    help="Ce qu'ajoute un tour : sorties d'outils, message.")
            opt_int("Tokens d'une requête chaude P95", "warm_p95", origin="warm")
        with c4:
            opt_float("Attente entre deux requêtes d'une session (s)", "session_gap", 0.0, step=1.0,
                      help="Outils et humain, moyenne : donne les sessions à rejouer et la capacité en sessions.")
            opt_int("Sessions en pause à garder en cache", "paused_sessions", min_value=0)
        st.radio("Mode", ["interactive", "batch"], horizontal=True, key="mode",
                 format_func=lambda m: {"interactive": "Interactif", "batch": "Batch"}[m])

        st.markdown("**Speculative decoding**")
        c1, c2, c3, c4 = st.columns(4)
        c1.checkbox("Méthode native du modèle", key="spec_enabled")
        with c2:
            opt_int("num_speculative_tokens (k)", "k", help="Vide : valeur recommandée pour le modèle.")
        c3.selectbox("draft_sample_method", ["défaut vLLM", "greedy", "probabilistic"], key="draft_sample_method")
        c4.checkbox("Tolérer une vérification lossy (FLy)", key="allow_lossy")

        st.markdown("**Contexte, arguments et ressources**")
        model = catalog.models[st.session_state["model_id"]]
        ext, native = model.context_extension, model.architecture.max_position_embeddings
        n = lambda v: f"{v:,}".replace(",", " ")  # noqa: E731
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            opt_int("max-model-len cible", "max_model_len",
                    help=f"Vide : contexte natif ({n(native)} tokens)." + (
                        f" Au-delà, extension {ext.method.upper()} jusqu'à {n(ext.max_model_len)}." if ext else ""))
        with c2:
            opt_float("Cœurs CPU par instance", "cpu_cores", 1.0, step=1.0,
                      help="Limite CPU du pod : plafonne --api-server-count. Vide : pas de plafond.")
        with c3:
            opt_float("Forcer gpu-memory-utilization", "gpu_util", 0.05, 1.0)
        with c4:
            opt_float("Réserve par GPU pour l'estimation du KV (Gio)", "reserve", 0.0, step=1.0)
        c1, c2 = st.columns(2)
        c1.text_area("Arguments vLLM supplémentaires (YAML)", key="extra_args_text", height=100,
                     placeholder="--enable-log-requests: true\n--enable-prompt-tokens-details: true",
                     help="Une ligne par flag, vérifiée contre le code source de la version. Remplace un flag généré.")
        c2.text_area("Variables d'environnement (YAML)", key="extra_env_text", height=100,
                     placeholder="VLLM_LOGGING_LEVEL: DEBUG")

        st.markdown("**Routage llm-d (EPP) entre les instances**")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Version d'llm-d", sorted(catalog.llmd_versions), key="llmd_version",
                     format_func=lambda v: catalog.llmd_versions[v].display_name)
        c2.radio("Index des préfixes", ["approximate", "precise"], horizontal=True, key="llmd_mode",
                 format_func=lambda m: {"approximate": "Approximatif", "precise": "Precise (KV events)"}[m])
        with c3:
            opt_int("Préfixe commun à toutes les sessions (tokens)", "llmd_shared_prefix", min_value=0,
                    help="System prompt + outils, en cache sur tous les pods. Vide : 0.")
        if st.session_state["llmd_mode"] == "precise":
            st.text_input("Endpoint ZMQ des KV events", key="llmd_kv_endpoint", placeholder="tcp://<service-epp>:5557")
        st.text_area("Paramètres llm-d forcés (YAML)", key="llmd_params_text", height=70,
                     placeholder="lruCapacityPerServer: 700000")
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, key) in zip((c1, c2, c3, c4), (("Poids queue", "llmd_w_queue"),
                                                        ("Poids kv-cache-utilization", "llmd_w_kv"),
                                                        ("Poids prefix-cache", "llmd_w_prefix"),
                                                        ("Poids no-hit-lru", "llmd_w_nohit"))):
            with col:
                opt_int(label, key, min_value=0, help="Vide : poids calculé par l'app.")

    st.divider()
    if st.session_state["isl_p50"] and st.session_state["osl_p50"]:
        next_button(1, "Voir les hypothèses et la config proposée")
    else:
        st.info("Renseignez au minimum ISL P50 et OSL P50 (ou importez une trace, ou chargez le cas de référence).")
    st.stop()


# --- Request and generation (steps ② to ⑤) ---------------------------------------------------------------------------

ss = st.session_state
if not (ss["isl_p50"] and ss["osl_p50"]):
    st.info("Renseignez au minimum ISL P50 et OSL P50 à l'étape ①.")
    st.button("← Revenir à ①", on_click=go, args=(0,))
    st.stop()


def parse_mapping(text: str, label: str) -> dict:
    """YAML mapping typed in a text area ({} when empty). Raises ValueError with a readable message."""
    if not text.strip():
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} : YAML invalide ({exc}).") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{label} : attendu un dictionnaire « clé: valeur », une entrée par ligne.")
    return {str(k): v for k, v in data.items()}


def pct(*keys: str) -> dict:
    return {p: ss[k] for p, k in zip(("p50", "p95", "p99"), keys) if ss[k] is not None}


def build_request() -> DeploymentRequest:
    perf = {"itl_p95_ms": ss["itl_measured"], "ttft_p95_ms": ss["ttft_measured"], "tpot_p95_ms": ss["tpot_measured"],
            "inflight_per_instance": ss["inflight_measured"], "sessions_per_instance": ss["sessions_measured"],
            "capacity_sessions_per_instance": ss["capacity_measured"]}
    observed = (ss["llmd_obs_block"], ss["llmd_obs_lru"], ss["llmd_obs_window"], ss["llmd_obs_hit"])
    return DeploymentRequest(
        gpu_id=ss["gpu_id"], gpus_per_instance=int(ss["gpus_per_instance"]), num_instances=int(ss["num_instances"]),
        model_id=ss["model_id"], vllm_version=ss["vllm_version"],
        workload={"usage_type": ss["usage_type"], "mode": ss["mode"], "isl": pct("isl_p50", "isl_p95", "isl_p99"),
                  "osl": pct("osl_p50", "osl_p95", "osl_p99"), "prefix_cache_hit_rate": ss["hit_rate"],
                  "cold_request_fraction": ss["cold_fraction"], "target_concurrency": ss["target_concurrency"],
                  "paused_sessions": ss["paused_sessions"],
                  "warm_uncached_tokens": pct("warm_p50", "warm_p95") if ss["warm_p50"] else None,
                  "session_gap_s": ss["session_gap"]},
        slo={"ttft_p95_ms": ss["ttft"], "itl_p95_ms": ss["itl"], "itl_metric": ss["itl_metric"]},
        allow_lossy=ss["allow_lossy"],
        spec={"enabled": ss["spec_enabled"], "num_speculative_tokens": ss["k"], "acceptance_length": ss["acceptance_length"],
              "draft_sample_method": None if ss["draft_sample_method"] == "défaut vLLM" else ss["draft_sample_method"]},
        observed_kv=({"kv_cache_tokens": ss["kv_tokens"], "gpu_memory_utilization": ss["kv_util"],
                      "block_size": ss["kv_block"]} if ss["kv_tokens"] else None),
        observed_perf=perf if any(v is not None for k, v in perf.items() if k != "sessions_per_instance") else None,
        gpu_memory_utilization=ss["gpu_util"], kv_estimate_reserve_gib_per_gpu=ss["reserve"],
        cpu_cores_per_instance=ss["cpu_cores"], max_model_len=ss["max_model_len"],
        extra_args=parse_mapping(ss["extra_args_text"], "Arguments supplémentaires"),
        extra_env={k: str(v) for k, v in parse_mapping(ss["extra_env_text"], "Variables d'environnement").items()},
        routing={
            "llmd_version": ss["llmd_version"], "prefix_mode": ss["llmd_mode"],
            "shared_prefix_tokens": ss["llmd_shared_prefix"],
            "kv_events_endpoint": (ss["llmd_kv_endpoint"] or None) if ss["llmd_mode"] == "precise" else None,
            "weights": {scorer: ss[key] for scorer, key in ROUTING_WEIGHT_KEYS.items() if ss[key] is not None},
            "parameters": parse_mapping(ss["llmd_params_text"], "Paramètres llm-d forcés"),
            "observed": ({"block_size_tokens": observed[0], "lru_capacity_per_server": observed[1],
                          "max_prefix_tokens_to_match": observed[2], "prefix_cache_hit_rate": observed[3]}
                         if any(v is not None for v in observed) else None),
        },
    )


try:
    request = build_request()
except ValueError as exc:  # pydantic's ValidationError is a ValueError
    if isinstance(exc, ValidationError):
        st.error("Entrées invalides :\n\n" + "\n".join(f"- {'.'.join(map(str, e['loc']))} : {e['msg']}"
                                                       for e in exc.errors()))
    else:
        st.error(str(exc))
    st.button("← Corriger à l'étape ①", on_click=go, args=(0,))
    st.stop()

try:
    result = generate(request, catalog)
except GenerationError as exc:
    st.error("\n\n".join(exc.errors))
    st.stop()
try:
    routing = generate_routing(result, catalog)
    routing_error = None
except GenerationError as exc:
    routing, routing_error = None, "\n\n".join(exc.errors)
epp_yaml = render_epp_config(routing, catalog) if routing is not None and not routing.blocking_errors else ""
analysis = result.analysis
k8s = None if result.blocking_errors else render.render_k8s_args(result, catalog)

for error in result.blocking_errors:
    st.error(error)


# --- ② Hypothèses et config ----------------------------------------------------------------------------------------------

if step == 1:
    st.subheader("Ce que l'app a supposé")
    st.caption("Les lignes « supposée » sont des hypothèses de l'app : corrigez à l'étape ① (ou ④ pour les mesures)"
               " celles qui ne correspondent pas à votre cas, la config se recalcule.")
    rows = report.hypotheses(request, result, ss["origins"])
    rows.sort(key=lambda h: h["origin"] != report.ASSUMED)
    st.dataframe([{"Hypothèse": h["label"], "Valeur": str(h["value"]), "Origine": h["origin"],
                   "Statut": CONFIDENCE_LABEL.get(h["status"], h["status"]), "Pourquoi": h["why"],
                   "Où corriger": h["where"]} for h in rows], width="stretch", hide_index=True)

    critical = [w for w in result.warnings if w.severity == "critical"]
    for w in critical:
        st.error(w.message)
    if k8s is None:
        st.stop()

    st.subheader("Configuration proposée")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("max-num-seqs", result.facts.max_num_seqs)
    c2.metric("max-num-batched-tokens", result.facts.max_num_batched_tokens)
    c3.metric("KV par instance", f"{result.facts.kv_tokens_per_instance:,}".replace(",", " ")
              if result.facts.kv_tokens_per_instance else "—",
              help={"startup_log": "Lu dans les logs", "estimate": "Estimation (à vérifier)"}.get(result.facts.kv_tokens_source))
    if analysis is not None:
        c4.metric("Verdict estimé", VERDICT_LABEL[analysis.verdict])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.code(k8s, language="yaml")
        st.download_button("Télécharger vllm-args.yaml", k8s, file_name="vllm-args.yaml")
    with c2:
        st.markdown("**Ligne de commande**")
        st.code(render.render_cli(result, catalog), language="bash")
        env_block = render.render_k8s_env(result)
        if env_block:
            st.markdown("**`env` du conteneur vLLM**")
            st.code(env_block, language="yaml")

    st.subheader("Pourquoi ces valeurs")
    st.dataframe([{"Paramètre": d.title, "Valeur": json.dumps(d.value, ensure_ascii=False)
                   if isinstance(d.value, dict) else str(d.value), "Émis": "oui" if d.emitted else "non",
                   "Statut": CONFIDENCE_LABEL[d.confidence], "En bref": first_sentence(d.justification)}
                  for d in result.decisions], width="stretch", hide_index=True)
    with st.expander("Détail de chaque paramètre (règle, formule, calcul, sources)"):
        for d in result.decisions:
            st.markdown(f"**{d.title}** · `{d.key}` = `{d.value}` · {CONFIDENCE_LABEL[d.confidence]}"
                        f" · règle `{d.rule_id}`" + (f" · défaut vLLM `{d.default}`" if d.default is not None else ""))
            st.write(d.justification)
            if d.formula or d.computation:
                st.caption(f"Formule : {d.formula or '—'} · Calcul : {d.computation or '—'}")
            if d.sources:
                st.caption("Sources : " + " · ".join(d.sources))

    if analysis is not None:
        st.subheader("Ce que l'app en attend")
        st.write(analysis.summary)
        checks_table(analysis, with_measure=False)
        if analysis.session_capacity is not None:
            st.markdown("**Capacité en sessions par instance**")
            capacity_block(analysis.session_capacity)

    st.subheader("Routage llm-d (EPP)")
    if routing_error:
        st.error(routing_error)
    elif routing.blocking_errors:
        for error in routing.blocking_errors:
            st.error(error)
    else:
        llmd_cat = catalog.llmd_versions[routing.llmd_version]
        st.caption(f"{result.facts.num_instances} instance(s) routées. À poser dans : {llmd_cat.placement.value}.")
        for warning in [x for x in routing.warnings if x.severity == "critical"]:
            st.error(warning.message)
        st.code(epp_yaml, language="yaml")
        st.download_button("Télécharger epp-config.yaml", epp_yaml, file_name="epp-config.yaml")
        vllm_extra = render_vllm_args(routing)
        if vllm_extra:
            st.markdown("**À ajouter aux args vLLM (mode precise)**")
            st.code(vllm_extra, language="yaml")
        with st.expander("📖 Comprendre les paramètres llm-d et leurs justifications"):
            st.markdown(
                "L'EPP choisit le pod vLLM de chaque requête : chaque **scorer** note les pods, l'EPP somme ces notes"
                " pondérées par les **poids** du profil et retient le pod le mieux noté. Le scorer de préfixe s'appuie sur"
                " un **index** des débuts de prompts envoyés à chaque pod, pour renvoyer une session là où son préfixe"
                " est en cache.")
            prefix_plugin = routing.plugins[0]["type"]  # the prefix plugin always comes first
            decided = {d.key: d.value for d in routing.decisions}
            st.dataframe([{"Paramètre": name, "C'est quoi": param.description,
                           "Défaut EPP": "—" if param.value is None else str(param.value),
                           "Proposé": str(decided.get(name, "—"))}
                          for name, param in llmd_cat.plugins[prefix_plugin].parameters.items()],
                         width="stretch", hide_index=True)
            st.dataframe([{"Scorer": scorer, "C'est quoi": llmd_cat.plugins[scorer].note or "",
                           "Défaut RHOAI": str(llmd_cat.default_profile.weights.get(scorer, "—")),
                           "Poids proposé": weight} for scorer, weight in routing.weights.items()],
                         width="stretch", hide_index=True)
            for d in routing.decisions:
                st.markdown(f"**{d.title}** = `{d.value}` · {CONFIDENCE_LABEL[d.confidence]}")
                st.write(d.justification)
            for a in routing.assumptions:
                st.warning(f"**Hypothèse — {a.label}** = `{a.value}`\n\n{a.reason}")
            st.markdown("**Vérifier après déploiement** : lignes attendues dans les logs de l'EPP (à comparer, puis"
                        " saisir à l'étape ④ avec le hit rate du pire pod).")
            st.code("\n".join(routing.expected_logs), language="text")

    others = [w for w in result.warnings if w.severity != "critical"]
    with st.expander(f"Autres avertissements ({len(others)}) et indicateurs"):
        show_warnings(others)
        st.dataframe([{"Indicateur": i.label, "Valeur": i.value, "Unité": i.unit or "", "Lecture": i.interpretation or "",
                       "Statut": CONFIDENCE_LABEL[i.confidence]} for i in result.indicators],
                     width="stretch", hide_index=True)
    st.divider()
    next_button(2, "Tester cette config")


# --- ③ Tester ------------------------------------------------------------------------------------------------------------

elif step == 2:
    if analysis is None or analysis.test_plan is None:
        st.info("Plan de test indisponible pour cette configuration.")
        st.stop()
    plan = analysis.test_plan
    st.subheader("Plan de test par paliers")
    st.write(plan.summary)
    sessions = request.workload.session_gap_s is not None
    st.dataframe([{
        "Palier": f"{'★ ' if s.is_generated_config else ''}{s.index}",
        "max-num-seqs": s.max_num_seqs,
        "max-num-batched-tokens": s.max_num_batched_tokens,
        "Requêtes en vol / instance": s.inflight_per_instance,
        **({"Sessions working-set / instance": "—" if s.sessions_per_instance is None
            else f"{s.sessions_per_instance[0]} – {s.sessions_per_instance[1]}",
            "KV des sessions": "—" if s.sessions_kv_pct is None else f"{s.sessions_kv_pct:g} %"} if sessions else {}),
        ("ITL p95 attendue" if request.slo.itl_metric == "itl" else "TPOT p95 attendu"):
            fmt_range(s.itl_ms if request.slo.itl_metric == "itl" else s.tpot_ms),
        "TTFT p95 attendu": fmt_range(s.ttft_ms),
        "KV": "—" if s.kv_usage_pct is None else f"{s.kv_usage_pct:g} %",
        "Attendu vs SLO": VERDICT_LABEL[s.expectation],
        "Si validé : instances (GPU)": "—" if s.instances_needed is None else f"{s.instances_needed} ({s.gpus_needed})",
        "Pourquoi": s.expectation_reason,
    } for s in plan.steps], width="stretch", hide_index=True)
    st.caption("★ : palier de la config générée. Plages = incertitude du modèle"
               + (f" (×{analysis.uncertainty[0]:g} à ×{analysis.uncertainty[1]:g})." if analysis.uncertainty else "."))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Protocole**")
        st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(plan.protocol, 1)))
    with c2:
        st.markdown("**Critères de passage d'un palier**")
        st.markdown("\n".join(f"- {c}" for c in plan.pass_criteria))
        for note in plan.notes:
            st.info(note)
    with st.expander("Vérifier les flags sur l'image et métriques à surveiller"):
        st.code(render.render_verify_command(result, catalog), language="bash")
        st.dataframe([{"Métrique": m.name, "Usage": m.usage} for m in catalog.workload.metrics],
                     width="stretch", hide_index=True)
        for test in catalog.workload.validation_tests:
            st.info(test)
    st.divider()
    next_button(3, "J'ai des mesures : recaler et ajuster")


# --- ④ Mesurer et ajuster ------------------------------------------------------------------------------------------------

elif step == 3:
    st.subheader("Importer un rejeu working-set (ws replay --out)")
    st.file_uploader("Résultat (.json)", type=["json"], key="run_file", on_change=import_run)
    message("run_message")
    profile = ss.get("run_profile")
    if profile is not None:
        st.selectbox("Palier de recalage", list(range(len(profile.steps))), key="run_step", on_change=apply_run,
                     format_func=lambda i: f"{profile.steps[i].sessions} sessions {'✅' if profile.steps[i].passed else '🛑'}",
                     help="Par défaut le plus grand palier validé, le plus proche de la limite.")
        st.dataframe(traces.run_rows(profile), hide_index=True, width="stretch")
        st.caption(f"working-set valide {profile.serves} sessions" + (f", échoue à {profile.fails_at}" if profile.fails_at
                                                                       else "") + ".")
        for note in profile.notes:
            st.caption(note)

    with st.expander("Ou saisir les mesures à la main (logs de démarrage, Prometheus)",
                     expanded=profile is None):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Logs de démarrage**")
            opt_int("GPU KV cache size (tokens)", "kv_tokens", help="Ligne « GPU KV cache size: N tokens ».")
            opt_float("gpu-memory-utilization lors de la mesure", "kv_util", 0.05, 1.0)
            opt_int("kv cache block size", "kv_block")
        with c2:
            st.markdown("**Latences mesurées**")
            opt_float("ITL / TPOT p95 (ms, métrique de la cible)", "itl_measured", 0.1, step=1.0)
            opt_float("TTFT p95 (ms)", "ttft_measured", 1.0, step=10.0)
            opt_float("TPOT p95 (ms, si la cible est l'ITL)", "tpot_measured", 0.1, step=1.0)
            opt_float("Requêtes en vol par instance pendant la mesure", "inflight_measured", 0.01, step=1.0,
                      help="Vide : mesure faite à la concurrence cible.")
        with c3:
            st.markdown("**Spec decoding et sessions**")
            opt_float("Longueur d'acceptation mesurée a", "acceptance_length", 1.0, step=0.1,
                      help="a = 1 + spec_decode_num_accepted_tokens / spec_decode_num_drafts")
            opt_float("Sessions par instance (palier mesuré)", "sessions_measured", 0.01, step=1.0)
            opt_float("Sessions par instance validées (capacité)", "capacity_measured", 0.0, step=1.0)
        st.markdown("**llm-d** (log « Prefix DataProducer initialized » et hit rate du pire pod)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            opt_int("blockSizeTokens (log)", "llmd_obs_block")
        with c2:
            opt_int("lruCapacityPerServer (log)", "llmd_obs_lru")
        with c3:
            opt_int("maxPrefixTokensToMatch (log)", "llmd_obs_window")
        with c4:
            opt_float("Hit rate, pire pod (0–1)", "llmd_obs_hit", 0.0, 1.0)

    if analysis is None:
        st.stop()
    measured = request.observed_perf is not None or request.observed_kv is not None
    st.subheader("Résultat après recalage" if measured else "Résultat (estimation non recalée)")
    if not measured:
        st.info("Importez un rejeu ou saisissez des mesures : le modèle se recale dessus et les estimations se resserrent.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Facteur decode", "—" if analysis.calibration is None else f"× {analysis.calibration:.2f}")
        c2.metric("Facteur prefill", "—" if analysis.calibration_prefill is None else f"× {analysis.calibration_prefill:.2f}")
        for note in analysis.notes:
            if any(word in note for word in ("recal", "Mesure faite", "conciliables", "extrapolée")):
                st.caption(note)
    checks_table(analysis, with_measure=True)
    if analysis.session_capacity is not None:
        st.subheader("Capacité en sessions par instance")
        capacity_block(analysis.session_capacity)
    st.subheader("Goulots et pistes")
    for b in analysis.bottlenecks:
        (st.error if b.status == "ko" else st.warning)(f"**{b.label}** — {b.explanation}")
    for i, r in enumerate(analysis.recommendations, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {RECO_LABEL[r.kind]} — {r.title}**")
            st.write(r.rationale)
            if r.changes:
                st.code("\n".join(f"{key}: {json.dumps(value, separators=(',', ':')) if isinstance(value, dict) else value}"
                                  for key, value in r.changes.items()), language="yaml")
    st.divider()
    next_button(4, "Exporter le rapport")


# --- ⑤ Exporter ----------------------------------------------------------------------------------------------------------

else:
    st.subheader("Exporter pour relecture")
    st.caption("Le rapport rassemble la demande, les hypothèses (avec leur origine), la config et ses justifications, ce"
               " que l'app en attend, les mesures et les limites connues de l'outil, avec les questions posées au"
               " relecteur. Envoyez le Markdown à un agent (ou une personne), le JSON à un outil.")
    st.text_input("Nom", key="scenario_name", placeholder="ex. Prod agentique B300")
    name = ss["scenario_name"] or None
    summary = ss.get("trace_summary")
    review = report.build_report(result, catalog, routing, epp_yaml or None, ss["origins"],
                                 summary[0] if summary else None, ss.get("run_profile"), name)
    markdown = report.render_markdown(review)
    base = slug(name) if name else "vllm-config"
    c1, c2, c3 = st.columns(3)
    c1.download_button("📝 Rapport de relecture (.md)", markdown, file_name=f"{base}-rapport.md", mime="text/markdown",
                       type="primary", width="stretch")
    c2.download_button("🧾 Rapport structuré (.json)", report.report_json(review), file_name=f"{base}-rapport.json",
                       mime="application/json", width="stretch")
    c3.download_button("💾 Scénario rechargeable (.yaml)", dump_scenario(request, name, result, k8s, epp_yaml or None),
                       file_name=f"{base}.yaml", mime="application/x-yaml", width="stretch",
                       help="À recharger à l'étape ① pour reprendre exactement ces entrées.")
    with st.expander("Aperçu du rapport", expanded=True):
        st.markdown(markdown)
