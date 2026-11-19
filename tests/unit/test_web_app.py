"""Smoke test of the Streamlit UI (headless, streamlit.testing): the five-step workflow."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "src" / "vllm_configurator" / "web" / "app.py")
STEPS = ["① Votre besoin", "② Hypothèses et config", "③ Tester", "④ Mesurer et ajuster", "⑤ Exporter"]


def app():
    return AppTest.from_file(APP, default_timeout=60).run()


def go(at, index):
    at.radio(key="step").set_value(STEPS[index]).run()
    assert not at.exception
    return at


def reference():
    at = app()
    at.button(key="load_reference").click().run()
    assert not at.exception
    return at


def codes(at):
    return [c.value for c in at.code]


def test_empty_form_asks_for_inputs():
    at = app()
    assert not at.exception
    assert any("ISL P50" in info.value for info in at.info)
    go(at, 1)
    assert any("ISL P50" in info.value for info in at.info)


def test_reference_case_goes_to_assumptions_and_config():
    at = reference()
    assert at.radio(key="step").value == STEPS[1]
    assert any("Ce que l'app a supposé" in h.value for h in at.subheader)
    hypotheses = next(df.value for df in at.dataframe if "Hypothèse" in df.value.columns)
    assert "supposée" in set(hypotheses["Origine"])
    yaml_block = next(c for c in codes(at) if c.startswith("args:"))
    assert '- "512"' in yaml_block and '{"max_cudagraph_capture_size":2048}' in yaml_block
    assert not [w for w in at.warning if "Session State" in w.value]


def test_values_survive_the_steps_and_the_plan_follows_the_target():
    at = reference()
    go(at, 0)
    at.number_input(key="itl").set_value(10.0).run()
    go(at, 2)
    assert any("Plan de test par paliers" in h.value for h in at.subheader)
    go(at, 0)
    assert at.number_input(key="itl").value == 10.0
    at.radio(key="itl_metric").set_value("tpot").run()
    go(at, 2)
    assert any("TPOT p95 attendu" in c for df in at.dataframe for c in df.value.columns)


def test_export_gives_the_review_report():
    at = go(reference(), 4)
    assert any("Consignes de relecture" in m.value for m in at.markdown)
    assert any("Hypothèses" in m.value for m in at.markdown)


def test_context_extension_and_extra_args_from_the_form():
    at = app()
    at.selectbox(key="model_id").set_value("qwen3.8-flash-next")
    at.number_input(key="isl_p50").set_value(200000)
    at.number_input(key="osl_p50").set_value(1000)
    at.number_input(key="max_model_len").set_value(1000000)
    at.text_area(key="extra_args_text").set_value("--enable-log-requests: true").run()
    go(at, 3)
    at.number_input(key="kv_tokens").set_value(20000000).run()
    go(at, 1)
    assert any(c.startswith("env:") and "VLLM_ALLOW_LONG_MAX_MODEL_LEN" in c for c in codes(at))
    assert any("--hf-overrides" in c and "--enable-log-requests" in c for c in codes(at) if c.startswith("args:"))


def test_invalid_extra_args_yaml_is_reported():
    at = go(reference(), 0)
    at.text_area(key="extra_args_text").set_value("- pas un dict").run()
    go(at, 1)
    assert any("Arguments supplémentaires" in e.value for e in at.error)


def test_app_uses_the_repository_package_even_if_a_stale_copy_is_loaded(tmp_path, monkeypatch):
    # Streamlit Community Cloud installs the project on the first deployment, then updates the repository without
    # reinstalling: an old copy of vllm_configurator can be imported before app.py.
    import sys

    stale = tmp_path / "vllm_configurator"
    stale.mkdir()
    (stale / "__init__.py").write_text("STALE = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    saved = {k: v for k, v in sys.modules.items() if k == "vllm_configurator" or k.startswith("vllm_configurator.")}
    for name in saved:
        monkeypatch.delitem(sys.modules, name)
    import vllm_configurator

    assert getattr(vllm_configurator, "STALE", False)
    reference()
    assert not getattr(sys.modules["vllm_configurator"], "STALE", False)


def test_llmd_config_is_part_of_the_proposal():
    at = reference()
    epp = next(c for c in codes(at) if c.startswith("apiVersion:"))
    assert "lruCapacityPerServer: 673639" in epp
    go(at, 0)
    at.radio(key="llmd_mode").set_value("precise").run()
    go(at, 1)
    assert any("endpoint ZMQ" in e.value for e in at.error)
    go(at, 0)
    at.text_input(key="llmd_kv_endpoint").set_value("tcp://epp:5557").run()
    go(at, 1)
    assert any("--kv-events-config" in c for c in codes(at))


def test_llmd_parameters_can_be_forced_and_are_explained():
    at = go(reference(), 0)
    at.text_area(key="llmd_params_text").set_value("lruCapacityPerServer: 700000").run()
    go(at, 1)
    epp = next(c for c in codes(at) if c.startswith("apiVersion:"))
    assert "lruCapacityPerServer: 700000" in epp and "forcé (calculé : 673639)" in epp
    assert any(e.label.startswith("📖 Comprendre") for e in at.expander)


def test_sessions_flow_through_test_and_measure_steps():
    at = go(reference(), 0)
    assert any(e.label.startswith("📼 Importer une trace") for e in at.expander)
    at.number_input(key="session_gap").set_value(6.0)
    at.number_input(key="warm_p50").set_value(1000)
    at.number_input(key="warm_p95").set_value(8000).run()
    go(at, 2)
    assert "Sessions working-set / instance" in [c for df in at.dataframe for c in df.value.columns]
    go(at, 3)
    assert any("Importer un rejeu working-set" in h.value for h in at.subheader)
    assert any("Capacité en sessions" in h.value for h in at.subheader)
    at.number_input(key="tpot_measured").set_value(20.0)
    at.number_input(key="inflight_measured").set_value(40.0).run()
    assert any("Résultat après recalage" in h.value for h in at.subheader)
