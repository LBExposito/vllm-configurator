"""Scenario files: YAML export / import of a deployment request."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vllm_configurator import render
from vllm_configurator.catalog import default_catalog
from vllm_configurator.domain import DeploymentRequest
from vllm_configurator.engine import generate
from vllm_configurator.scenario import KIND, ScenarioError, dump_scenario, load_scenario, request_to_form

REFERENCE = Path(__file__).resolve().parents[1] / "golden" / "reference_case" / "input.yaml"


@pytest.fixture
def request_():
    data = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    data["slo"] = {"itl_p95_ms": 40, "ttft_p95_ms": 2000}
    data["observed_perf"] = {"itl_p95_ms": 31.5}
    data["workload"]["target_concurrency"] = 800
    return DeploymentRequest(**data)


def test_round_trip_keeps_every_input(request_):
    text = dump_scenario(request_, "Prod agentique")
    loaded, name = load_scenario(text)
    assert loaded == request_
    assert name == "Prod agentique"
    doc = yaml.safe_load(text)
    assert doc["kind"] == KIND and doc["version"] == 1
    assert doc["request"]["slo"] == {"ttft_p95_ms": 2000.0, "itl_p95_ms": 40.0, "itl_metric": "itl"}


def test_generated_snapshot_is_informative_only(request_):
    result = generate(request_)
    text = dump_scenario(request_, None, result, render.render_k8s_args(result, default_catalog()))
    doc = yaml.safe_load(text)
    assert "--max-num-seqs" in doc["generated"]["args"]
    assert doc["generated"]["slo_verdict"] == result.analysis.verdict
    doc["generated"]["args"] = ["--bogus"]  # modifié à la main : ignoré à l'import
    loaded, _ = load_scenario(yaml.safe_dump(doc, allow_unicode=True))
    assert loaded == request_


def test_bare_request_is_accepted():
    loaded, name = load_scenario(REFERENCE.read_text(encoding="utf-8"))
    assert loaded.model_id == "deepseek-v4.1-flash-1017" and name is None


def test_form_mapping_rebuilds_the_same_request(request_):
    form = request_to_form(request_)
    assert form["itl"] == 40.0 and form["itl_measured"] == 31.5 and form["target_concurrency"] == 800
    assert form["draft_sample_method"] == "probabilistic"
    assert form["kv_tokens"] == 86225834


@pytest.mark.parametrize("text, message", [
    ("- a\n- b\n", "objet YAML"),
    ("kind: autre/chose\nrequest: {}\n", "kind inattendu"),
    (f"kind: {KIND}\nversion: 99\nrequest: {{}}\n", "plus récente"),
    (f"kind: {KIND}\nversion: 1\n", "request"),
    ("gpu_id: b300\n", "Scénario invalide"),
    ("a: [", "YAML invalide"),
])
def test_invalid_files_are_rejected_with_a_reason(text, message):
    with pytest.raises(ScenarioError, match=message):
        load_scenario(text)
