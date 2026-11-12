"""Rules engine behaviour, including version-dependent rules."""

from __future__ import annotations

import pytest

from vllm_configurator import render
from vllm_configurator.domain import DeploymentRequest, GenerationError
from vllm_configurator.engine import generate

BASE = {
    "gpu_id": "b300", "gpus_per_instance": 4, "num_instances": 2, "model_id": "deepseek-v4.1-flash-1017",
    "vllm_version": "0.33.0",
    "workload": {"usage_type": "agentic", "isl": {"p50": 24000, "p95": 75000}, "osl": {"p50": 250}},
    "spec": {"num_speculative_tokens": 3},
    "observed_kv": {"kv_cache_tokens": 86225834, "gpu_memory_utilization": 0.8},
}


def req(**overrides) -> DeploymentRequest:
    data = {**BASE}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return DeploymentRequest(**data)


def values(result) -> dict:
    return {d.key: d.value for d in result.decisions}


def indicators(result) -> dict:
    return {i.key: i.value for i in result.indicators}


def categories(result) -> set:
    return {w.category for w in result.warnings}


class TestVersionDependentRules:
    def test_dspark_budget_accounting_0_33(self):
        assert indicators(generate(req()))["decode_tokens_per_seq"] == 4

    def test_dspark_budget_accounting_0_32_reserves_draft_slots(self):
        # 0.32.0 : 1 + k (vérification) + k − 1 (slots de drafting DSpark) = 6 pour k = 3.
        result = generate(req(vllm_version="0.32.0"))
        assert indicators(result)["decode_tokens_per_seq"] == 6
        assert "dspark_active_seq_cap" not in indicators(result)

    def test_fly_unavailable_before_0_33(self):
        result = generate(req(vllm_version="0.32.0", allow_lossy=True))
        assert values(result)["speculative_config"]["rejection_sample_method"] == "standard"
        assert "lossy" not in categories(result)

    def test_adaptive_long_prefill_flag_emitted_since_0_31(self):
        for version in ("0.31.0", "0.32.0", "0.33.0"):
            assert values(generate(req(vllm_version=version)))["long_prefill_token_threshold_adaptive"] is True

    def test_draft_lm_head_quantization_hint_only_in_0_33(self):
        rule_ids = lambda r: {w.rule_id for w in r.warnings}  # noqa: E731
        assert "R-SPEC-05" in rule_ids(generate(req()))
        assert "R-SPEC-05" not in rule_ids(generate(req(vllm_version="0.32.0")))

    def test_unverified_behavior_surfaces_as_to_verify(self):
        result = generate(req(vllm_version="0.32.0"))
        assert any(w.category == "to_verify" and "frontend.rejects_unknown_flags" in w.message for w in result.warnings)

    def test_unknown_version_is_rejected(self):
        with pytest.raises(GenerationError, match="0.99.0"):
            generate(req(vllm_version="0.99.0"))


class TestSpeculative:
    def test_fly_enabled_when_lossy_allowed_with_critical_warning_for_agentic(self):
        result = generate(req(allow_lossy=True))
        config = values(result)["speculative_config"]
        assert config["rejection_sample_method"] == "fly"
        assert config["fly_window_size"] == 2  # min(6, k − 1)
        assert any(w.category == "lossy" and w.severity == "critical" for w in result.warnings)

    def test_fixed_k_overrides_input(self):
        result = generate(req(model_id="mistral-large-4", spec={"num_speculative_tokens": 5}))
        assert values(result)["speculative_config"]["num_speculative_tokens"] == 3
        assert any("impose k = 3" in w.message for w in result.warnings)

    def test_recommended_k_when_not_provided(self):
        result = generate(req(spec={"num_speculative_tokens": None}))
        assert values(result)["speculative_config"]["num_speculative_tokens"] == 5
        assert values(result)["compilation_config"] == {"max_cudagraph_capture_size": 3072}  # 6 × 512

    def test_mtp_model(self):
        result = generate(req(model_id="qwen3.8-flash-next", observed_kv={"kv_cache_tokens": 10_000_000}, spec={"num_speculative_tokens": None}))
        config = values(result)["speculative_config"]
        assert config["method"] == "mtp" and "enable_adaptive_verification" not in config
        assert indicators(result)["decode_tokens_per_seq"] == 4

    def test_spec_disabled(self):
        result = generate(req(spec={"enabled": False}))
        assert "--speculative-config" not in render.argv(result)
        assert values(result)["compilation_config"] == {"max_cudagraph_capture_size": 512}
        assert values(result)["gpu_memory_utilization"] == 0.92


class TestScheduler:
    def test_batch_mode_budget(self):
        result = generate(req(workload={"mode": "batch"}))
        assert values(result)["max_num_batched_tokens"] == 16384
        assert values(result)["long_prefill_token_threshold"] == 4096

    def test_target_concurrency_caps_max_num_seqs(self):
        result = generate(req(workload={"target_concurrency": 300}))
        assert values(result)["max_num_seqs"] == 150
        assert values(result)["compilation_config"] == {"max_cudagraph_capture_size": 600}

    def test_concurrency_above_start_point_warns(self):
        result = generate(req(workload={"target_concurrency": 4000}))
        assert values(result)["max_num_seqs"] == 512
        assert "itl" in categories(result)

    def test_kv_limited_caps_max_num_seqs(self):
        result = generate(req(observed_kv={"kv_cache_tokens": 2_400_000}))
        assert values(result)["max_num_seqs"] == 100  # 2,4 M / 24 000

    def test_capture_clamped_to_budget(self):
        result = generate(req(workload={"usage_type": "chat"}, spec={"num_speculative_tokens": 7}))
        assert values(result)["compilation_config"]["max_cudagraph_capture_size"] <= values(result)["max_num_batched_tokens"]


class TestKvCapacity:
    def test_estimate_without_logs_is_flagged(self):
        result = generate(req(observed_kv=None))
        assert result.facts.kv_tokens_source == "estimate"
        assert result.facts.kv_tokens_per_instance > 0
        assert sum(w.category == "to_verify" for w in result.warnings) >= 2  # octets/token et HBM
        assert any(a.key == "kv_reserve" for a in result.assumptions)

    def test_unknown_kv_blocks_generation(self):
        result = generate(req(model_id="glm-5.3-flash", observed_kv=None))
        assert result.blocking_errors and "GPU KV cache size" in result.blocking_errors[0]

    def test_qwen_per_sequence_state_warning(self):
        result = generate(req(model_id="qwen3.8-flash-next", observed_kv=None))
        assert any("état fixe" in w.message for w in result.warnings)

    def test_offload_suggested_when_kv_too_small(self):
        result = generate(req(observed_kv={"kv_cache_tokens": 2_400_000},
                              workload={"target_concurrency": 200, "paused_sessions": 200}))
        assert "kv_limited" in categories(result)
        assert "--kv-offloading-size" not in render.argv(result)


class TestSupport:
    def test_mistral_large_4_requires_custom_image(self):
        result = generate(req(model_id="mistral-large-4", observed_kv={"kv_cache_tokens": 3_000_000}))
        messages = [w.message for w in result.warnings if w.severity == "critical"]
        assert any("vllm/vllm-openai:mistral-large-4" in m for m in messages)

    def test_context_overflow(self):
        result = generate(req(model_id="qwen3.8-flash-next", observed_kv={"kv_cache_tokens": 10_000_000},
                              workload={"isl": {"p50": 24000, "p95": 75000, "p99": 300000}}))
        assert any(w.rule_id == "R-KV-07" and w.severity == "critical" for w in result.warnings)


class TestRender:
    def test_cli_and_k8s_contain_the_same_flags(self):
        result = generate(req())
        cli = render.render_cli(result)
        k8s = render.render_k8s_args(result)
        for token in render.argv(result):
            if token.startswith("--"):
                assert token in cli and token in k8s

    def test_verify_command_lists_every_emitted_flag(self):
        result = generate(req())
        command = render.render_verify_command(result)
        assert command.startswith("vllm serve --help=all")
        for token in render.argv(result):
            if token.startswith("--"):
                assert token[2:] in command

    def test_json_values_are_compact(self):
        args = render.argv(generate(req()))
        assert '{"max_cudagraph_capture_size":2048}' in args
