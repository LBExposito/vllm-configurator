"""Context extension (YaRN via --hf-overrides) and user-supplied extra vLLM arguments / env variables."""

from __future__ import annotations

import json

import pytest

from vllm_configurator import render
from vllm_configurator.domain import DeploymentRequest
from vllm_configurator.engine import generate
from vllm_configurator.scenario import dump_scenario, load_scenario, request_to_form

QWEN = {
    "gpu_id": "b300", "gpus_per_instance": 4, "model_id": "qwen3.8-flash-next", "vllm_version": "0.33.0",
    "workload": {"usage_type": "long_summary", "isl": {"p50": 200000, "p99": 900000}, "osl": {"p50": 1000}},
    "observed_kv": {"kv_cache_tokens": 20_000_000, "gpu_memory_utilization": 0.9},
}


def run(**overrides):
    return generate(DeploymentRequest(**{**QWEN, **overrides}))


def values(result) -> dict:
    return {d.key: d for d in result.decisions}


def messages(result, category) -> list[str]:
    return [w.message for w in result.warnings if w.category == category]


class TestContextExtension:
    def test_qwen_1m_follows_the_model_card(self):
        result = run(max_model_len=1_000_000)
        d = values(result)
        assert d["max_model_len"].emitted and d["max_model_len"].value == 1_000_000
        rope = d["hf_overrides"].value["text_config"]["rope_parameters"]
        # Bloc complet de la model card, factor 4.0 pour 1 000 000 tokens.
        assert rope == {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn",
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                        "original_max_position_embeddings": 262144, "factor": 4.0}
        assert result.env == {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
        cli = render.render_cli(result)
        assert cli.startswith("VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve")
        assert "--max-model-len 1000000" in cli and "--hf-overrides '{\"text_config\"" in cli
        assert 'value: "1"' in render.render_k8s_env(result)
        assert any("YARN active" in m for m in messages(result, "context"))
        assert not [w for w in result.warnings if w.category == "unsupported" and "contexte servi" in w.message]

    @pytest.mark.parametrize("target, factor", [(524_288, 2.0), (1_000_000, 4.0), (300_000, 1.5)])
    def test_factor_matches_the_target_length(self, target, factor):
        rope = values(run(max_model_len=target))["hf_overrides"].value["text_config"]["rope_parameters"]
        assert rope["factor"] == factor

    def test_without_target_the_native_context_is_kept_and_the_extension_suggested(self):
        result = run()
        assert not values(result)["max_model_len"].emitted
        assert "hf_overrides" not in values(result)
        assert result.env == {}
        assert any("Extension possible jusqu'à" in w.message for w in result.warnings)  # séparateur : espace fine

    def test_shorter_target_is_emitted_without_override(self):
        result = run(max_model_len=131072, workload={**QWEN["workload"], "isl": {"p50": 20000}})
        assert values(result)["max_model_len"].emitted and "hf_overrides" not in values(result)
        assert result.env == {}

    def test_beyond_the_extension_is_refused(self):
        result = run(max_model_len=2_000_000)
        assert not values(result)["max_model_len"].emitted
        assert any("extension maximale" in m for m in messages(result, "context"))

    def test_model_without_extension_needs_a_manual_override(self):
        data = {**QWEN, "model_id": "deepseek-v4.1-flash-1017", "max_model_len": 2_000_000}
        result = generate(DeploymentRequest(**data))
        assert any("sans extension connue" in m for m in messages(result, "context"))


class TestExtraArgs:
    def test_valid_flags_are_emitted_in_their_own_section(self):
        result = run(extra_args={"--enable-log-requests": True, "served-model-name": "qwen-long"})
        k8s = render.render_k8s_args(result)
        assert "# --- Arguments supplémentaires ---" in k8s
        assert "  - --enable-log-requests" in k8s and '  - --served-model-name\n  - "qwen-long"' in k8s

    def test_unknown_flag_is_dropped_with_a_critical_warning(self):
        result = run(extra_args={"--definitely-not-a-flag": 1})
        assert "--definitely-not-a-flag" not in render.render_cli(result)
        assert any(w.severity == "critical" and "absent de vLLM" in w.message for w in result.warnings)

    def test_extra_flag_replaces_the_generated_one(self):
        result = run(extra_args={"--gpu-memory-utilization": 0.85})
        assert render.argv(result).count("--gpu-memory-utilization") == 1
        assert "0.85" in render.argv(result)
        assert "remplacé par l'argument supplémentaire" in values(result)["gpu_memory_utilization"].omitted_reason

    def test_json_values_from_dict_or_string(self):
        override = {"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 2.0}}}
        for value in (override, json.dumps(override)):
            result = run(max_model_len=500_000, extra_args={"--hf-overrides": value})
            argv = render.argv(result)
            assert argv.count("--hf-overrides") == 1
            assert json.loads(argv[argv.index("--hf-overrides") + 1]) == override
            assert result.env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"  # override manuel : env posée quand même

    def test_engine_driven_flag_warns(self):
        result = run(extra_args={"--max-num-seqs": 64})
        assert any("max-num-seqs forcé" in m for m in messages(result, "extra_args"))

    def test_extra_env_is_merged(self):
        result = run(max_model_len=1_000_000, extra_env={"VLLM_LOGGING_LEVEL": "DEBUG"})
        assert result.env == {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "VLLM_LOGGING_LEVEL": "DEBUG"}


def test_scenario_round_trip_keeps_context_and_extra_args():
    request = DeploymentRequest(**{**QWEN, "max_model_len": 1_000_000,
                                   "extra_args": {"--enable-log-requests": True}, "extra_env": {"A": "1"}})
    loaded, _ = load_scenario(dump_scenario(request, "qwen-1m", generate(request)))
    assert loaded == request
    form = request_to_form(request)
    assert form["max_model_len"] == 1_000_000 and "--enable-log-requests: true" in form["extra_args_text"]
