"""Section 3 formulas, checked against the numbers of docs/vllm-config-knowledge-base.md."""

import pytest

from vllm_configurator import formulas as f

KV_REF = 86_225_834  # « GPU KV cache size » of the reference case (§1)


class TestRequiredRatio:
    """§3.1, table « Scénario (B = 16384, S = 128) »."""

    def test_without_spec_decoding(self):
        assert f.required_ratio(16384, 128, 1, 1) == pytest.approx(127, abs=0.5)

    def test_mtp_k1(self):
        d = f.decode_tokens_per_seq(1, extra_draft_slots=0)
        assert d == 2
        assert f.required_ratio(16384, 128, d, 1.8) == pytest.approx(70, abs=0.5)

    @pytest.mark.parametrize(("a", "expected"), [(2.5, 51), (3.0, 42), (3.5, 36)])
    def test_dspark_k3_as_written_in_the_document(self, a, expected):
        # The document assumes d ≈ 1 for DSpark in 0.33; the vLLM 0.33.0 source code says otherwise (see below).
        assert f.required_ratio(16384, 128, 1, a) == pytest.approx(expected, abs=0.5)

    @pytest.mark.parametrize(("a", "expected"), [(2.5, 49.6), (3.0, 41.3), (3.5, 35.4)])
    def test_dspark_k3_with_verification_tokens_charged(self, a, expected):
        # vLLM 0.33.0: d = 1 + k (verification tokens), no extra drafting slot for DSpark.
        d = f.decode_tokens_per_seq(3, extra_draft_slots=0)
        assert d == 4
        assert f.required_ratio(16384, 128, d, a) == pytest.approx(expected, abs=0.05)

    def test_reference_case_final_config(self):
        # §7: B = 8192, S = 512, DSpark k = 3, a = 3 → ratio needed 4.
        assert f.required_ratio(8192, 512, 4, 3.0) == pytest.approx(4.0)

    def test_rejects_zero_sequences(self):
        with pytest.raises(ValueError):
            f.required_ratio(8192, 0, 1, 1)


class TestDecodeTokens:
    def test_no_spec(self):
        assert f.decode_tokens_per_seq(0, extra_draft_slots=0) == 1

    def test_pr_59468_charged_6_instead_of_10(self):
        # PR #59468, DSpark K = 5: « charged 6 tokens instead of 10 ».
        before = f.decode_tokens_per_seq(5, f.eval_slot_expression("k-1", 5))
        after = f.decode_tokens_per_seq(5, f.eval_slot_expression("0", 5))
        assert (before, after) == (10, 6)
        # « each decode schedules 1 + K = 6 tokens, so the 512-token budget fills at 85 ».
        assert 512 // after == 85

    @pytest.mark.parametrize(("expr", "k", "expected"), [("0", 3, 0), ("1", 3, 1), ("k", 3, 3), ("k-1", 3, 2)])
    def test_slot_expressions(self, expr, k, expected):
        assert f.eval_slot_expression(expr, k) == expected

    def test_unknown_slot_expression(self):
        with pytest.raises(ValueError):
            f.eval_slot_expression("2k", 3)


class TestEffectiveRatio:
    def test_copilot_median_call(self):
        # §6: median prompt 68K of which 63K cached, median output 247.
        assert f.effective_ratio(68_000, 63 / 68, 247) == pytest.approx(5_000 / 247)

    def test_tracelab_aggregate(self):
        # §6: 2.34 B new tokens / 186.9 M generated ≈ 12.5.
        assert f.effective_ratio(2.34e9, 0.0, 186.9e6) == pytest.approx(12.5, abs=0.05)

    def test_rejects_invalid_hit_rate(self):
        with pytest.raises(ValueError):
            f.effective_ratio(1000, 1.5, 100)


class TestKvConcurrency:
    """§3.2 et log de démarrage du §1."""

    def test_p50(self):
        assert f.kv_concurrency(KV_REF, 24_000) == pytest.approx(3_600, rel=0.01)

    def test_p95(self):
        assert f.kv_concurrency(KV_REF, 75_000) == pytest.approx(1_150, rel=0.01)

    def test_startup_log_max_concurrency(self):
        # « Maximum concurrency for 1,048,576 tokens per request: 82.23x ».
        assert f.max_concurrency_for_length(KV_REF, 1_048_576) == pytest.approx(82.23, abs=0.005)


class TestCudagraphCapture:
    """§3.3, k = 3."""

    @pytest.mark.parametrize(("seqs", "expected"), [(128, 512), (256, 1024), (384, 1536), (512, 2048)])
    def test_k3(self, seqs, expected):
        assert f.cudagraph_capture_size(3, seqs) == expected

    def test_clamped_to_budget_and_multiple_of_k_plus_1(self):
        assert f.clamp_capture_size(2048, 8192, 4) == 2048
        assert f.clamp_capture_size(9000, 8192, 4) == 8192
        assert f.clamp_capture_size(9000, 8190, 6) == 8190  # B300 preset of the DeepSeek recipe (k = 5)
        assert f.clamp_capture_size(2050, 8192, 4) == 2048


def test_verified_tokens_per_step():
    # §3.4: 512 sequences, k = 3 → about 2,000 verified tokens per step.
    assert f.verified_tokens_per_step(512, 3) == 2048


def test_lru_capacity_per_server():
    # §3.5: 86,225,834 / 128 ≈ 673,000.
    assert f.lru_capacity_per_server(KV_REF, 128) == pytest.approx(673_000, rel=0.001)


def test_tokens_per_expert():
    # §3.6 with DeepSeek V4.1 Flash (config.json: 384 routed experts, top-6).
    assert f.tokens_per_expert(128, 6, 384) == 2
    assert f.tokens_per_expert(8192, 6, 384) == 128


class TestLongPrefillThreshold:
    """§2.5, vérifié dans vllm/v1/core/sched/scheduler.py (0.31 à 0.33)."""

    def test_pr_58459_example_adaptive(self):
        # Budget 8192, threshold 2048, two requests: 4096 each with the adaptive floor.
        assert f.long_prefill_effective_threshold(2048, 8192, 0, 2, single_request_bypass=True, adaptive=True) == 4096

    def test_pr_58459_example_without_flag(self):
        # Without --long-prefill-token-threshold-adaptive (default in 0.33), the fixed threshold applies.
        assert f.long_prefill_effective_threshold(2048, 8192, 0, 2, single_request_bypass=True, adaptive=False) == 2048

    def test_pr_57951_single_request(self):
        assert f.long_prefill_effective_threshold(2048, 8192, 0, 1, single_request_bypass=True, adaptive=False) == 0

    def test_decodes_are_in_the_divisor(self):
        # 512 sequences in decode + 1 prefill: 8192 // 513 = 15 < 2048, so the fixed threshold applies.
        assert f.long_prefill_effective_threshold(2048, 8192, 512, 1, single_request_bypass=True, adaptive=True) == 2048

    def test_disabled(self):
        assert f.long_prefill_effective_threshold(0, 8192, 3, 3, single_request_bypass=True, adaptive=True) == 0


def test_dspark_active_seq_cap_pr_59468():
    # PR #59468: budget 512, K = 5, cap of 102 running requests (num_lookahead_tokens = K).
    assert f.dspark_active_seq_cap(512, 5) == 102
    assert f.dspark_active_seq_cap(8192, 4) == 2048


class TestKvEstimate:
    def test_replicated_mla(self):
        tokens = f.kv_tokens_estimate(
            hbm_bytes_per_gpu=100e9, gpu_memory_utilization=0.9, weights_bytes=200e9, tensor_parallel_size=4,
            reserve_bytes_per_gpu=10e9, bytes_per_token=1000, replicated_across_tp=True, num_kv_heads=1,
        )
        assert tokens == 30_000_000  # (90 − 50 − 10) GB / 1000 B

    def test_gqa_split_over_kv_heads(self):
        tokens = f.kv_tokens_estimate(
            hbm_bytes_per_gpu=100e9, gpu_memory_utilization=1.0, weights_bytes=0, tensor_parallel_size=4,
            reserve_bytes_per_gpu=0, bytes_per_token=12288, replicated_across_tp=False, num_kv_heads=2,
        )
        assert tokens == int(100e9 // 6144)  # 2 KV heads: divided by min(TP, 2)

    def test_no_free_memory(self):
        assert f.kv_tokens_estimate(
            hbm_bytes_per_gpu=10e9, gpu_memory_utilization=0.5, weights_bytes=100e9, tensor_parallel_size=1,
            reserve_bytes_per_gpu=0, bytes_per_token=100, replicated_across_tp=True, num_kv_heads=1,
        ) == 0


class TestDistributions:
    """Distributions used by the SLO analysis (tokens to prefill, acceptance length)."""

    def test_normal_ppf(self):
        assert f.normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert f.normal_ppf(0.975) == pytest.approx(1.95996, abs=1e-4)
        assert f.normal_ppf(0.01) == pytest.approx(-2.32635, abs=1e-4)

    def test_lognormal_quantiles_match_the_percentiles(self):
        q = f.lognormal_quantile_fn(24000, 75000, 666000)
        assert q(0.5) == pytest.approx(24000)
        assert q(0.95) == pytest.approx(75000, rel=1e-3)
        assert q(0.99) == pytest.approx(666000, rel=1e-3)  # tail stretched up to the observed P99
        assert f.lognormal_quantile_fn(24000, 75000, 666000, cap=500000)(0.995) == 500000

    def test_uncached_distribution_keeps_the_hit_rate(self):
        q = f.lognormal_quantile_fn(24000, 75000)
        points, hit = f.uncached_distribution(q, 0.957, 0.02, 1.0)
        assert sum(p for p, _, _ in points) == pytest.approx(1.0)
        assert hit == pytest.approx(0.957, abs=1e-3)
        mean_isl = sum(p * isl for p, _, isl in points)
        assert sum(p * u for p, u, _ in points) == pytest.approx(0.043 * mean_isl, rel=1e-3)
        # Cold requests make the tail: much heavier than ISL P95 × (1 − hit).
        assert f.quantile_of_points(points, 0.99) > 5 * 75000 * 0.043

    def test_cold_requests_beyond_the_hit_rate_lower_the_effective_hit(self):
        q = f.lognormal_quantile_fn(24000, 75000)
        _, hit = f.uncached_distribution(q, 0.957, 0.2, 1.0)
        assert hit == pytest.approx(0.8, abs=0.01)

    def test_no_cache_prefills_every_prompt(self):
        q = f.lognormal_quantile_fn(24000, 75000)
        points, hit = f.uncached_distribution(q, 0.0, 0.0, 1.0)
        assert hit == 0.0 and all(u == isl for _, u, isl in points)

    def test_collapse_keeps_probability_and_mean(self):
        points, _ = f.uncached_distribution(f.lognormal_quantile_fn(24000, 75000), 0.9, 0.05, 1.0)
        merged = f.collapse(points)
        assert len(merged) < len(points)
        assert sum(p for p, _, _ in merged) == pytest.approx(1.0)
        assert sum(p * u for p, u, _ in merged) == pytest.approx(sum(p * u for p, u, _ in points), rel=1e-9)

    def test_interpolated_quantile_has_no_jump_at_slice_edges(self):
        slices = [(0.5, 10.0), (0.5, 20.0)]
        assert f.quantile_interpolated(slices, 0.25) == 10.0
        assert f.quantile_interpolated(slices, 0.5) == pytest.approx(15.0)
        assert f.quantile_interpolated(slices, 0.75) == 20.0

    def test_acceptance_length(self):
        assert f.tokens_per_step(0.0, 3) == 1.0
        assert f.tokens_per_step(1.0, 3) == 4.0
        assert f.tokens_per_step(0.5, 3) == pytest.approx(1.875)
        low = f.acceptance_quantile(3.0, 3, 0.05, 8)
        assert 1.0 < low < 3.0
        assert f.acceptance_quantile(3.0, 3, 0.05, 1000) == pytest.approx(3.0, rel=0.05)  # little dispersion
        assert f.acceptance_quantile(1.0, 0, 0.05, 8) == 1.0
