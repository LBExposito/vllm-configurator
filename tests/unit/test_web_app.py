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


def test_reference_case_shows_the_slo_interpretation():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    at.number_input(key="itl").set_value(10.0).run()
    assert not at.exception
    assert any("Interprétation SLO" in e.value for e in at.error)
    assert any("Goulots" in h.value for h in at.subheader)


def test_scenario_export_is_offered():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.session_state["scenario_name"] == "Cas de référence (§1)"
    assert any(e.label.startswith("💾 Scénarios") for e in at.sidebar.expander)
