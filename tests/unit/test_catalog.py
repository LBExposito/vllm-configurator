"""Catalog consistency: every catalog file loads and every emitted name exists in the vLLM source."""

import pytest

from vllm_configurator.catalog import load_catalog


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_cross_validation_is_clean(catalog):
    assert catalog.validate() == []


def test_every_version_has_extracted_flags(catalog):
    assert set(catalog.vllm_versions) == set(catalog.extracted)


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_behaviors_used_by_the_engine_exist(catalog, version):
    required = {
        "long_prefill.single_request_bypass", "long_prefill.adaptive_floor", "spec_decode.verification_tokens_in_budget",
        "spec_decode.extra_draft_slots", "spec_decode.dspark_active_seq_cap", "spec_decode.fly",
        "spec_decode.adaptive_verification", "spec_decode.draft_lm_head_quantization",
        "cudagraph.default_max_capture_size", "cudagraph.capped_by_max_num_batched_tokens", "cudagraph.multiple_of_k_plus_1",
        "max_model_len.yarn_factor_extends_derived", "max_model_len.allow_long_env", "hf_overrides.nested_merge",
    }
    assert required <= set(catalog.vllm_versions[version].behaviors)


def test_speculative_keys_emitted_by_the_engine_exist_in_0_33(catalog):
    keys = set(catalog.extracted["0.33.0"].speculative_config_keys)
    assert {"method", "num_speculative_tokens", "draft_sample_method", "rejection_sample_method", "fly_window_size",
            "fly_entropy_threshold", "fly_entropy_top_k", "enable_adaptive_verification"} <= keys
    assert "max_cudagraph_capture_size" in catalog.extracted["0.33.0"].compilation_config_keys


def test_fly_only_from_0_33(catalog):
    assert "fly" in catalog.extracted["0.33.0"].rejection_sample_methods
    assert "fly" not in catalog.extracted["0.32.0"].rejection_sample_methods
    assert catalog.vllm_versions["0.32.0"].behaviors["spec_decode.fly"].value == {"available": False}


def test_dspark_draft_slots_change_in_0_33(catalog):
    slots = {v: catalog.vllm_versions[v].behaviors["spec_decode.extra_draft_slots"].value["dspark"] for v in catalog.vllm_versions}
    assert slots == {"0.31.0": "k-1", "0.32.0": "k-1", "0.33.0": "0"}


def test_models_pin_a_revision_and_their_architecture_status(catalog):
    for model in catalog.models.values():
        assert len(model.hf_revision) == 40
        assert model.hf_revision in str(model.architecture.source), model.id


def test_mistral_large_4_is_not_in_the_0_33_registry(catalog):
    arch = catalog.models["mistral-large-4"].architecture.hf_architecture
    assert arch not in catalog.extracted["0.33.0"].architectures


def test_unverified_kv_values_never_claim_verified(catalog):
    for model in catalog.models.values():
        if model.kv_cache.bytes_per_token is None:
            assert "logs" in model.kv_cache.method, model.id
