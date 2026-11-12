"""llm-d EPP target: routing rules derived from the vLLM facts (docs/vllm-config-knowledge-base.md §3.5, §9)."""

from pathlib import Path

import pytest
import yaml

from vllm_configurator.catalog import default_catalog
from vllm_configurator.domain import DeploymentRequest
from vllm_configurator.engine import generate
from vllm_configurator.llmd import generate_routing, render_epp_config, render_vllm_args
from vllm_configurator.scenario import dump_scenario, load_scenario, request_to_form
from vllm_configurator.targets import TARGETS

REFERENCE = Path(__file__).resolve().parents[1] / "golden" / "reference_case" / "input.yaml"


def reference(**changes) -> DeploymentRequest:
    data = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    for key, value in changes.items():
        data[key] = value
    return DeploymentRequest(**data)


def routing_for(request: DeploymentRequest):
    return generate_routing(generate(request))


def test_catalog_has_the_rhoai_epp():
    catalog = default_catalog()
    assert "rhoai-3.5" in catalog.llmd_versions
    assert catalog.validate() == []
    llmd = catalog.llmd_versions["rhoai-3.5"]
    assert llmd.default_profile.weights == {"queue-scorer": 2, "kv-cache-utilization-scorer": 2,
                                            "prefix-cache-scorer": 3, "no-hit-lru-scorer": 2}


def test_reference_case_matches_the_proposed_epp_config():
    # §9.4 : blockSizeTokens 128, lruCapacityPerServer ≈ 86 225 834 / 128, autoTune off, poids 2 / 1 / 5 / 2.
    r = routing_for(reference())
    assert not r.blocking_errors
    assert r.decision("blockSizeTokens").value == 128
    assert r.decision("lruCapacityPerServer").value == 86_225_834 // 128
    assert r.decision("autoTune").value is False
    assert r.decision("maxPrefixTokensToMatch").value == 131072
    assert r.decision("maxPrefixBlocksToMatch").value == 1024
    assert r.weights == {"queue-scorer": 2, "kv-cache-utilization-scorer": 1, "prefix-cache-scorer": 5,
                         "no-hit-lru-scorer": 2}


def test_rendered_epp_config_is_valid_yaml():
    r = routing_for(reference())
    doc = yaml.safe_load(render_epp_config(r))
    assert doc["kind"] == "EndpointPickerConfig"
    producer = doc["plugins"][0]
    assert producer["type"] == "approx-prefix-cache-producer"
    assert producer["parameters"]["lruCapacityPerServer"] == 673639
    profile = doc["schedulingProfiles"][0]["plugins"]
    assert {"pluginRef": "prefix-cache-scorer", "weight": 5} in profile
    assert profile[-1] == {"pluginRef": "max-score-picker"}
    assert render_vllm_args(r) == ""


def qwen(**changes) -> DeploymentRequest:
    return reference(model_id="qwen3.8-flash-next", spec={"enabled": False}, **changes)


def test_small_vllm_block_is_raised_to_the_epp_minimum():
    r = routing_for(qwen(observed_kv={"kv_cache_tokens": 86225834, "block_size": 16}))
    assert r.decision("blockSizeTokens").value == 64
    assert r.decision("lruCapacityPerServer").value == 86225834 // 64


def test_low_reuse_workload_lowers_the_prefix_weight():
    data = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    data["workload"]["prefix_cache_hit_rate"] = 0.2
    r = routing_for(DeploymentRequest(**data))
    assert r.weights["prefix-cache-scorer"] == 2


def test_shared_prefix_reduces_the_value_of_affinity():
    data = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    data["workload"]["prefix_cache_hit_rate"] = 0.6
    alone = routing_for(DeploymentRequest(**data))
    data["routing"] = {"shared_prefix_tokens": 20000}
    shared = routing_for(DeploymentRequest(**data))
    gain = {i.key: i.value for i in alone.indicators}["prefill_gain"]
    gain_shared = {i.key: i.value for i in shared.indicators}["prefill_gain"]
    assert gain_shared < gain


def test_saturated_pods_raise_the_queue_weight():
    data = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    data["workload"]["target_concurrency"] = 1000  # 500 par pod pour max-num-seqs 512
    r = routing_for(DeploymentRequest(**data))
    assert r.weights["queue-scorer"] == 3


def test_forced_weight_wins():
    r = routing_for(reference(routing={"weights": {"prefix-cache-scorer": 4}}))
    assert r.weights["prefix-cache-scorer"] == 4
    assert "Forcé à 4" in r.decision("weight:prefix-cache-scorer").justification


def test_single_instance_is_flagged():
    r = routing_for(reference(num_instances=1))
    assert any(w.rule_id == "llmd.instances" for w in r.warnings)


def test_undersized_deployed_index_is_critical():
    # §9.3 : 31 250 blocs × 64 ≈ 2M tokens suivis pour 86M réels.
    r = routing_for(reference(routing={"observed": {"block_size_tokens": 16, "lru_capacity_per_server": 31250}}))
    messages = [w for w in r.warnings if w.rule_id == "llmd.observed"]
    assert any(w.severity == "critical" and "oublié" in w.message for w in messages)
    assert any("écrase" in w.message for w in messages)


def test_measured_hit_rate_gap_is_explained():
    r = routing_for(reference(routing={"observed": {"prefix_cache_hit_rate": 0.5}}))
    assert any("Hit rate mesuré" in w.message for w in r.warnings)


def test_precise_mode_needs_an_endpoint_then_emits_kv_events():
    blocked = routing_for(reference(routing={"prefix_mode": "precise"}))
    assert blocked.blocking_errors and render_epp_config(blocked) == ""
    r = routing_for(reference(routing={"prefix_mode": "precise", "kv_events_endpoint": "tcp://epp:5557"}))
    assert not r.blocking_errors
    assert r.decision("tokenProcessorConfig.blockSize").value == 128
    assert "precise-prefix-cache-scorer" in r.weights and "no-hit-lru-scorer" not in r.weights
    assert r.vllm_args["--kv-events-config"]["endpoint"] == "tcp://epp:5557"
    assert "--kv-events-config" in render_vllm_args(r)
    doc = yaml.safe_load(render_epp_config(r))
    assert doc["plugins"][0]["parameters"] == {"tokenProcessorConfig": {"blockSize": 128}}


def test_unknown_vllm_block_size():
    precise = {"prefix_mode": "precise", "kv_events_endpoint": "tcp://epp:5557"}
    blocked = routing_for(qwen(observed_kv={"kv_cache_tokens": 20000000}, routing=precise))
    assert any("block size" in e for e in blocked.blocking_errors)
    approx = routing_for(qwen(observed_kv={"kv_cache_tokens": 20000000}))
    assert approx.decision("blockSizeTokens").value == 64
    assert any(a.key == "blockSizeTokens" for a in approx.assumptions)


def test_unknown_llmd_version_is_rejected():
    from vllm_configurator.domain import GenerationError

    with pytest.raises(GenerationError):
        routing_for(reference(routing={"llmd_version": "nope"}))


def test_target_registry_exposes_llmd_epp():
    r = TARGETS["llmd_epp"](reference())
    assert r.decision("lruCapacityPerServer").value == 673639


def test_routing_round_trips_through_scenarios():
    request = reference(routing={"shared_prefix_tokens": 8000, "weights": {"queue-scorer": 3},
                                 "observed": {"prefix_cache_hit_rate": 0.9}})
    result = generate(request)
    text = dump_scenario(request, "llm-d", result, None, render_epp_config(generate_routing(result)))
    loaded, _ = load_scenario(text)
    assert loaded.routing == request.routing
    assert yaml.safe_load(text)["generated"]["llmd_epp"]["kind"] == "EndpointPickerConfig"
    form = request_to_form(loaded)
    assert form["llmd_shared_prefix"] == 8000 and form["llmd_w_queue"] == 3 and form["llmd_obs_hit"] == 0.9


def test_forced_parameters_replace_the_computed_values():
    r = routing_for(reference(routing={"parameters": {"lruCapacityPerServer": 700000, "maxPrefixTokensToMatch": 262144}}))
    assert r.decision("lruCapacityPerServer").value == 700000
    assert "calculé : 673639" in r.decision("lruCapacityPerServer").justification
    params = yaml.safe_load(render_epp_config(r))["plugins"][0]["parameters"]
    assert params["lruCapacityPerServer"] == 700000 and params["maxPrefixTokensToMatch"] == 262144
    # 262 144 tokens / 128 ≠ 1 024 blocs : incohérence signalée.
    assert any("maxPrefixBlocksToMatch" in w.message for w in r.warnings)


def test_forced_parameters_are_checked():
    r = routing_for(reference(routing={"parameters": {"lruCapacityPerServer": 31250, "nope": 1, "autoTune": "oui"}}))
    critical = [w.message for w in r.warnings if w.severity == "critical"]
    assert any("'nope' inconnu" in m for m in critical)
    assert any("autoTune attend true ou false" in m for m in critical)
    assert any("oublié" in m for m in critical)
    assert r.decision("autoTune").value is False


def test_precise_block_size_mismatch_is_critical():
    r = routing_for(reference(routing={"prefix_mode": "precise", "kv_events_endpoint": "tcp://epp:5557",
                                       "parameters": {"tokenProcessorConfig.blockSize": 64}}))
    assert yaml.safe_load(render_epp_config(r))["plugins"][0]["parameters"] == {"tokenProcessorConfig": {"blockSize": 64}}
    assert any(w.severity == "critical" and "hashes" in w.message for w in r.warnings)


def test_every_prefix_parameter_is_explained():
    for plugin in default_catalog().llmd_versions["rhoai-3.5"].plugins.values():
        for param in plugin.parameters.values():
            assert len(param.description) > 40
