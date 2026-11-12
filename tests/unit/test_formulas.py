"""Section 3 formulas, checked against the numbers of docs/vllm-config-knowledge-base.md."""

import pytest

from vllm_configurator import formulas as f

KV_REF = 86_225_834  # « GPU KV cache size » du cas de référence (§1)


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
        # Le document suppose d ≈ 1 pour DSpark en 0.33 ; le code source de vLLM 0.33.0 l'infirme (voir ci-dessous).
        assert f.required_ratio(16384, 128, 1, a) == pytest.approx(expected, abs=0.5)

    @pytest.mark.parametrize(("a", "expected"), [(2.5, 49.6), (3.0, 41.3), (3.5, 35.4)])
    def test_dspark_k3_with_verification_tokens_charged(self, a, expected):
        # vLLM 0.33.0 : d = 1 + k (tokens de vérification), 0 slot de drafting supplémentaire pour DSpark.
        d = f.decode_tokens_per_seq(3, extra_draft_slots=0)
        assert d == 4
        assert f.required_ratio(16384, 128, d, a) == pytest.approx(expected, abs=0.05)

    def test_reference_case_final_config(self):
        # §7 : B = 8192, S = 512, DSpark k = 3, a = 3 → ratio nécessaire 4.
        assert f.required_ratio(8192, 512, 4, 3.0) == pytest.approx(4.0)

    def test_rejects_zero_sequences(self):
        with pytest.raises(ValueError):
            f.required_ratio(8192, 0, 1, 1)


class TestDecodeTokens:
    def test_no_spec(self):
        assert f.decode_tokens_per_seq(0, extra_draft_slots=0) == 1

    def test_pr_59468_charged_6_instead_of_10(self):
        # PR #59468, DSpark K = 5 : « charged 6 tokens instead of 10 ».
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
        # §6 : prompt médian 68K dont 63K en cache, sortie médiane 247.
        assert f.effective_ratio(68_000, 63 / 68, 247) == pytest.approx(5_000 / 247)

    def test_tracelab_aggregate(self):
        # §6 : 2,34 Md de tokens nouveaux / 186,9 M générés ≈ 12,5.
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
        assert f.clamp_capture_size(9000, 8190, 6) == 8190  # préréglage B300 de la recette DeepSeek (k = 5)
        assert f.clamp_capture_size(2050, 8192, 4) == 2048


def test_verified_tokens_per_step():
    # §3.4 : 512 séquences, k = 3 → environ 2 000 tokens vérifiés par step.
    assert f.verified_tokens_per_step(512, 3) == 2048


def test_lru_capacity_per_server():
    # §3.5 : 86 225 834 / 128 ≈ 673 000.
    assert f.lru_capacity_per_server(KV_REF, 128) == pytest.approx(673_000, rel=0.001)


def test_tokens_per_expert():
    # §3.6 avec DeepSeek V4.1 Flash (config.json : 384 experts routés, top-6).
    assert f.tokens_per_expert(128, 6, 384) == 2
    assert f.tokens_per_expert(8192, 6, 384) == 128


class TestLongPrefillThreshold:
    """§2.5, vérifié dans vllm/v1/core/sched/scheduler.py (0.31 à 0.33)."""

    def test_pr_58459_example_adaptive(self):
        # Budget 8192, seuil 2048, deux requêtes : 4096 chacune avec le plancher adaptatif.
        assert f.long_prefill_effective_threshold(2048, 8192, 0, 2, single_request_bypass=True, adaptive=True) == 4096

    def test_pr_58459_example_without_flag(self):
        # Sans --long-prefill-token-threshold-adaptive (défaut en 0.33), le seuil fixe s'applique.
        assert f.long_prefill_effective_threshold(2048, 8192, 0, 2, single_request_bypass=True, adaptive=False) == 2048

    def test_pr_57951_single_request(self):
        assert f.long_prefill_effective_threshold(2048, 8192, 0, 1, single_request_bypass=True, adaptive=False) == 0

    def test_decodes_are_in_the_divisor(self):
        # 512 séquences en decode + 1 prefill : 8192 // 513 = 15 < 2048, donc le seuil fixe s'applique.
        assert f.long_prefill_effective_threshold(2048, 8192, 512, 1, single_request_bypass=True, adaptive=True) == 2048

    def test_disabled(self):
        assert f.long_prefill_effective_threshold(0, 8192, 3, 3, single_request_bypass=True, adaptive=True) == 0


def test_dspark_active_seq_cap_pr_59468():
    # PR #59468 : budget 512, K = 5, plafond de 102 requêtes running (num_lookahead_tokens = K).
    assert f.dspark_active_seq_cap(512, 5) == 102
    assert f.dspark_active_seq_cap(8192, 4) == 2048


class TestKvEstimate:
    def test_replicated_mla(self):
        tokens = f.kv_tokens_estimate(
            hbm_bytes_per_gpu=100e9, gpu_memory_utilization=0.9, weights_bytes=200e9, tensor_parallel_size=4,
            reserve_bytes_per_gpu=10e9, bytes_per_token=1000, replicated_across_tp=True, num_kv_heads=1,
        )
        assert tokens == 30_000_000  # (90 − 50 − 10) Go / 1000 o

    def test_gqa_split_over_kv_heads(self):
        tokens = f.kv_tokens_estimate(
            hbm_bytes_per_gpu=100e9, gpu_memory_utilization=1.0, weights_bytes=0, tensor_parallel_size=4,
            reserve_bytes_per_gpu=0, bytes_per_token=12288, replicated_across_tp=False, num_kv_heads=2,
        )
        assert tokens == int(100e9 // 6144)  # 2 têtes KV : divisé par min(TP, 2)

    def test_no_free_memory(self):
        assert f.kv_tokens_estimate(
            hbm_bytes_per_gpu=10e9, gpu_memory_utilization=0.5, weights_bytes=100e9, tensor_parallel_size=1,
            reserve_bytes_per_gpu=0, bytes_per_token=100, replicated_across_tp=True, num_kv_heads=1,
        ) == 0
