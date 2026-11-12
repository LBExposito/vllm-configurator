"""Smoke test of the Streamlit UI (headless, streamlit.testing)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "src" / "vllm_configurator" / "web" / "app.py")


def test_empty_form_asks_for_inputs():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("ISL P50" in info.value for info in at.info)


def test_reference_case_renders_the_config():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    yaml_block = next(code.value for code in at.code if code.value.startswith("args:"))
    assert '- "512"' in yaml_block and '{"max_cudagraph_capture_size":2048}' in yaml_block
    assert not [w for w in at.warning if "Session State" in w.value]


def test_reference_case_shows_the_test_plan():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    at.number_input(key="itl").set_value(10.0).run()
    assert not at.exception
    assert any("Paliers de test" in i.value for i in at.info)
    assert any("Plan de test par paliers" in h.value for h in at.subheader)
    at.radio(key="itl_metric").set_value("tpot").run()
    assert not at.exception


def test_scenario_export_is_offered():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.session_state["scenario_name"] == "Cas de référence (§1)"
    assert any(e.label.startswith("💾 Scénarios") for e in at.sidebar.expander)


def test_context_extension_and_extra_args_from_the_form():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key="model_id").set_value("qwen3.8-flash-next")
    at.number_input(key="isl_p50").set_value(200000)
    at.number_input(key="osl_p50").set_value(1000)
    at.number_input(key="kv_tokens").set_value(20000000)
    at.number_input(key="max_model_len").set_value(1000000)
    at.text_area(key="extra_args_text").set_value("--enable-log-requests: true").run()
    assert not at.exception
    codes = [c.value for c in at.code]
    assert any(c.startswith("env:") and "VLLM_ALLOW_LONG_MAX_MODEL_LEN" in c for c in codes)
    assert any("--hf-overrides" in c and "--enable-log-requests" in c for c in codes if c.startswith("args:"))


def test_invalid_extra_args_yaml_is_reported():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    at.text_area(key="extra_args_text").set_value("- pas un dict").run()
    assert not at.exception
    assert any("Arguments supplémentaires" in e.value for e in at.error)
