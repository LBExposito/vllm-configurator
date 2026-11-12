"""SLO analysis: latency model, verdict, bottleneck classification and corrective recommendations."""

from __future__ import annotations

import math

import pytest

from vllm_configurator import perf
from vllm_configurator.domain import DeploymentRequest
from vllm_configurator.engine import generate

# Cas de référence (§1) : DeepSeek V4.1 Flash, 2 pods TP4 de B300, KV lu dans les logs.
BASE = {
    "gpu_id": "b300", "gpus_per_instance": 4, "num_instances": 2, "model_id": "deepseek-v4.1-flash-1017",
    "vllm_version": "0.33.0",
    "workload": {"usage_type": "agentic", "isl": {"p50": 24000, "p95": 75000}, "osl": {"p50": 4000}},
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


def analysis(**overrides):
    return generate(req(**overrides)).analysis


def checks(a) -> dict:
    return {c.key: c for c in a.checks}


def bottlenecks(a) -> list[str]:
    return [b.key for b in a.bottlenecks]


def kinds(a) -> list[str]:
    return [r.kind for r in a.recommendations]


class TestPerfModel:
    def test_experts_hit_fraction(self):
        assert perf.experts_hit_fraction(1, 6, 384) == pytest.approx(6 / 384)
        assert perf.experts_hit_fraction(2048, 6, 384) == pytest.approx(1.0, abs=1e-9)
        assert perf.experts_hit_fraction(0, 6, 384) == 0.0

    def test_md1_wait(self):
        assert perf.md1_wait(0.5, 1.0) == pytest.approx(0.5)
        assert math.isinf(perf.md1_wait(1.0, 1.0))

    def test_moe_step_grows_sublinearly_with_sequences(self):
        a = analysis()
        model_note = a.notes[0]
        assert "544 G paramètres d'experts routés" in model_note
        assert a.step_breakdown["Borne du step chargé"] in ("mémoire", "calcul")


class TestVerdict:
    def test_reference_case_without_slo_is_a_capacity_verdict(self):
        a = analysis()
        assert a.verdict == "ok"
        assert checks(a)["itl"].status == "unknown"  # pas de cible
        assert checks(a)["kv"].status == "ok"
        assert kinds(a)[0] == "none"

    def test_reference_case_meets_loose_slo(self):
        a = analysis(slo={"itl_p95_ms": 50, "ttft_p95_ms": 2000})
        assert a.verdict == "ok"
        assert "devraient être tenus" in a.summary

    def test_full_load_with_short_agentic_outputs_saturates_the_prefill(self):
        # OSL 250 (médiane agentique publiée) à 512 séquences : ratio effectif 4,1 ≥ 4,0 nécessaire (§3.1).
        a = analysis(workload={"osl": {"p50": 250}}, slo={"ttft_p95_ms": 2000})
        assert checks(a)["prefill_load"].status == "ko"
        assert bottlenecks(a)[0] == "prefill"

    def test_cold_resume_is_reported_but_not_in_the_verdict(self):
        a = analysis(slo={"itl_p95_ms": 50, "ttft_p95_ms": 2000})
        cold = checks(a)["ttft_cold"]
        assert cold.in_verdict is False
        assert cold.estimate > checks(a)["ttft"].estimate

    def test_measured_value_wins_over_the_estimate(self):
        a = analysis(slo={"itl_p95_ms": 30}, observed_perf={"itl_p95_ms": 45})
        itl = checks(a)["itl"]
        assert itl.measured == 45 and itl.status == "ko"
        assert a.calibration and itl.estimate == pytest.approx(45, rel=0.01)  # modèle recalé sur la mesure

    def test_no_latency_model_without_performance_data(self):
        from vllm_configurator.catalog import load_catalog

        catalog = load_catalog()
        model = catalog.models["deepseek-v4.1-flash-1017"]
        stripped = model.model_copy(update={"performance": None})
        catalog.models["deepseek-v4.1-flash-1017"] = stripped
        try:
            a = generate(req(slo={"itl_p95_ms": 30}), catalog).analysis
        finally:
            catalog.models["deepseek-v4.1-flash-1017"] = model
        assert checks(a)["itl"].status == "unknown"
        assert a.step_breakdown is None
        assert any("Modèle de latence indisponible" in n for n in a.notes)


class TestBottlenecks:
    def test_prefill_saturation(self):
        # Sorties courtes : beaucoup de requêtes finissent par step, le prefill non caché dépasse le budget.
        a = analysis(workload={"target_concurrency": 1500, "osl": {"p50": 250}}, slo={"itl_p95_ms": 25, "ttft_p95_ms": 1500})
        assert a.verdict == "ko"
        assert bottlenecks(a)[0] == "prefill"
        assert checks(a)["prefill_load"].status == "ko"
        capacity = next(r for r in a.recommendations if r.kind == "capacity")
        assert capacity.changes["instances"] > 2

    def test_kv_capacity(self):
        a = analysis(observed_kv={"kv_cache_tokens": 5_000_000, "gpu_memory_utilization": 0.8},
                     workload={"target_concurrency": 600, "osl": {"p50": 250}}, slo={"ttft_p95_ms": 2000})
        assert "kv_capacity" in bottlenecks(a)
        titles = " ".join(r.title for r in a.recommendations)
        assert "--gpu-memory-utilization" in titles and "DP-attention" in titles
        tune = next(r for r in a.recommendations if r.kind == "param" and "--max-num-seqs" in r.changes)
        # Moins de séquences : le KV redescend sous le seuil, la capture suit (k + 1) × max_num_seqs.
        assert tune.changes["--max-num-seqs"] < 208
        assert tune.changes["--compilation-config"]["max_cudagraph_capture_size"] == 4 * tune.changes["--max-num-seqs"]

    def test_itl_floor_cannot_be_fixed_by_more_instances(self):
        a = analysis(workload={"target_concurrency": 100}, slo={"itl_p95_ms": 5})
        assert "itl_floor" in bottlenecks(a)
        assert "architecture" in kinds(a)

    def test_weight_bound_decode_prefers_wider_tp_to_hundreds_of_instances(self):
        a = analysis(workload={"target_concurrency": 800, "osl": {"p50": 250}}, slo={"itl_p95_ms": 30},
                     observed_perf={"itl_p95_ms": 45})
        assert bottlenecks(a)[0] == "decode"
        wider = next(r for r in a.recommendations if r.changes.get("--tensor-parallel-size") == 8)
        scale = next(r for r in a.recommendations if r.kind == "capacity" and "--tensor-parallel-size" not in r.changes)
        assert wider.changes["GPU"] < scale.changes["GPU"]

    def test_parameter_fix_without_extra_gpu_when_one_exists(self):
        # 500 séquences par instance, sorties courtes : le prefill sature le budget de 8192 ; 12288 suffit.
        a = analysis(workload={"target_concurrency": 1000, "osl": {"p50": 250}}, slo={"ttft_p95_ms": 2000, "itl_p95_ms": 40})
        assert a.verdict == "ko" and bottlenecks(a)[0] == "prefill"
        tune = a.recommendations[0]
        assert tune.kind == "param"
        assert tune.changes["--max-num-batched-tokens"] > 8192
        assert "--max-num-seqs" not in tune.changes  # pas de file d'attente ajoutée

def test_every_catalog_model_gets_an_analysis():
    for model_id, kv in (("glm-5.3-flash", 40_000_000), ("qwen3.8-flash-next", 20_000_000), ("mistral-large-4", 30_000_000)):
        data = {**BASE, "model_id": model_id, "observed_kv": {"kv_cache_tokens": kv, "gpu_memory_utilization": 0.9},
                "spec": {}, "slo": {"itl_p95_ms": 40, "ttft_p95_ms": 3000}}
        a = generate(DeploymentRequest(**data)).analysis
        assert a.verdict in ("ok", "at_risk", "ko")
        assert a.step_breakdown is not None
