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

    def test_erlang_c(self):
        assert perf.erlang_c(1, 0.5) == pytest.approx(0.5)  # M/M/1 : P(attente) = ρ
        assert perf.erlang_c(2, 1.0) == pytest.approx(1 / 3)
        assert perf.erlang_c(3, 3.0) == 1.0

    def test_mgc_wait_grows_with_the_service_variance(self):
        p_wait, scale = perf.mgc_wait(0.5, 1, 1.0, 0.0)
        assert p_wait * scale == pytest.approx(0.5)  # M/D/1 : ρ / (2 (1 − ρ)) × service
        _, heavy = perf.mgc_wait(0.5, 1, 1.0, 20.0)
        assert heavy == pytest.approx(21 * scale)  # facteur (1 + SCV)
        _, wait_c = perf.mgc_wait(0.5, 4, 1.0, 0.0)
        assert wait_c < scale  # plusieurs prefills de front : moins d'attente
        assert math.isinf(perf.mgc_wait(1.0, 1, 1.0, 0.0)[1])

    def test_quantile_of_sum(self):
        points = [(0.5, 1.0), (0.5, 3.0)]
        assert perf.quantile_of_sum(0.5, points, 0.0, 0.0) == pytest.approx(2.0)  # interpolation entre les tranches
        # W exponentielle de moyenne 1, toujours présente : P(W > x) = 0,05 à x = ln 20.
        assert perf.quantile_of_sum(0.95, [(1.0, 0.0)], 1.0, 1.0) == pytest.approx(math.log(20), rel=1e-3)
        assert perf.quantile_of_sum(0.95, points, 0.5, 2.0) > perf.quantile_of_sum(0.95, points, 0.0, 0.0)

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
        a = analysis(slo={"itl_p95_ms": 150, "ttft_p95_ms": 2000})
        assert a.verdict == "ok"
        assert "devraient être tenus" in a.summary

    def test_full_load_with_short_agentic_outputs_saturates_the_prefill(self):
        # OSL 250 (médiane agentique publiée) à 512 séquences : ratio effectif 4,1 ≥ 4,0 nécessaire (§3.1).
        a = analysis(workload={"osl": {"p50": 250}}, slo={"ttft_p95_ms": 2000})
        assert checks(a)["prefill_load"].status == "ko"
        assert bottlenecks(a)[0] == "prefill"

    def test_cold_resume_is_reported_but_not_in_the_verdict(self):
        a = analysis(slo={"itl_p95_ms": 150, "ttft_p95_ms": 2000})
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
        # Les paliers ne dépassent pas ce que le KV loge (90 % au contexte moyen) ; la capture suit (k + 1) × S.
        assert all(s.kv_usage_pct <= 90 for s in a.test_plan.steps)
        assert all(s.capture_size == min(4 * s.max_num_seqs, s.max_num_batched_tokens) for s in a.test_plan.steps)

    def test_itl_floor_cannot_be_fixed_by_more_instances(self):
        a = analysis(workload={"target_concurrency": 100}, slo={"itl_p95_ms": 5})
        assert "itl_floor" in bottlenecks(a)
        assert "architecture" in kinds(a)

    def test_weight_bound_decode_prefers_wider_tp_to_hundreds_of_instances(self):
        a = analysis(workload={"target_concurrency": 1600}, slo={"itl_p95_ms": 30}, observed_perf={"itl_p95_ms": 45})
        assert bottlenecks(a)[0] == "decode"
        wider = next(r for r in a.recommendations if r.changes.get("--tensor-parallel-size") == 8)
        scale = next(r for r in a.recommendations if r.kind == "capacity" and "--tensor-parallel-size" not in r.changes)
        assert wider.changes["GPU"] < scale.changes["GPU"]

    def test_parameter_fix_without_extra_gpu_when_one_exists(self):
        # 500 séquences par instance, sorties courtes : le prefill sature le budget de 8192 ; 12288 suffit, même dans
        # le cas pessimiste (×2,5) avec des cibles lâches (le step chargé d'un budget plus haut allonge l'ITL).
        a = analysis(workload={"target_concurrency": 1000, "osl": {"p50": 250}}, slo={"ttft_p95_ms": 5000, "itl_p95_ms": 300})
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



class TestRangesAndMetric:
    def test_uncalibrated_estimates_are_ranges(self):
        a = analysis(slo={"itl_p95_ms": 50, "ttft_p95_ms": 2000})
        itl = checks(a)["itl"]
        assert a.uncertainty == (0.8, 2.5)
        # Le multiplicateur porte sur le GPU ; l'attente du frontal (GIL) s'y ajoute sans être multipliée, et croît un peu
        # quand les steps sont plus rapides (plus de trafic) : d'où quelques pourcents d'écart.
        assert itl.estimate_low == pytest.approx(itl.estimate * 0.8, rel=0.03)
        assert itl.estimate_high == pytest.approx(itl.estimate * 2.5, rel=0.03)

    def test_calibration_tightens_the_range(self):
        a = analysis(slo={"itl_p95_ms": 50}, observed_perf={"itl_p95_ms": 30})
        assert a.uncertainty == (0.85, 1.15)

    def test_tpot_p95_divides_the_mean_step_by_the_acceptance_of_slow_requests(self):
        a = analysis(slo={"itl_p95_ms": 50, "itl_metric": "tpot"})
        tpot = checks(a)["itl"]
        assert tpot.label == "TPOT p95"
        # a moyen = 3 avec DSpark k = 3, mais ~2,2 pour les 5 % de requêtes les moins acceptées.
        step = a.step_breakdown["Step moyen (ms)"]
        assert step / 3 < tpot.estimate < step / 2
        assert any("les 5 % de requêtes les moins acceptées" in n for n in a.notes)

    def test_spec_decoding_lowers_tpot_less_than_the_mean_acceptance(self):
        on = checks(analysis(slo={"itl_p95_ms": 50, "itl_metric": "tpot"}))["itl"].estimate
        off = checks(analysis(slo={"itl_p95_ms": 50, "itl_metric": "tpot"}, spec={"enabled": False}))["itl"].estimate
        # Drafter, vérification de k + 1 tokens et requêtes peu acceptées : gain réel bien en deçà de a = 3.
        assert off / 3 < on < off

    def test_mean_context_from_percentiles(self):
        a = analysis(workload={"isl": {"p50": 23000, "p95": 76600, "p99": 666000}})
        assert any("ISL 35" in n for n in a.notes)  # moyenne ≈ 35 452, bien au-delà du P50

    def test_kv_is_rescaled_to_the_generated_memory_fraction(self):
        # KV lu à 0,8, config au défaut 0,92 (vLLM profile la mémoire des CUDA graphs) : plus de mémoire libre.
        a = analysis(observed_kv={"kv_cache_tokens": 86_225_864, "gpu_memory_utilization": 0.8})
        assert any(n.startswith("KV ramené") for n in a.notes)


class TestTestPlan:
    def test_plan_brackets_the_expected_limit(self):
        a = analysis(workload={"target_concurrency": 1700, "isl": {"p50": 23000, "p95": 76600, "p99": 666000},
                               "osl": {"p50": 4500, "p95": 11500, "p99": 24000}, "prefix_cache_hit_rate": 0.87},
                     slo={"itl_p95_ms": 50, "ttft_p95_ms": 60000}, spec={"enabled": False})
        plan = a.test_plan
        seqs = [s.max_num_seqs for s in plan.steps]
        assert seqs == sorted(seqs) and 2 <= len(seqs) <= 8
        assert max(seqs) <= 850  # inutile de tester au-delà des requêtes en vol par instance
        assert any(s.is_generated_config for s in plan.steps)
        assert plan.steps[0].expectation == "ok"
        assert all(s.instances_needed == -(-1700 // s.max_num_seqs) for s in plan.steps)
        assert "requêtes en vol" in plan.summary
        assert any("vllm:inter_token_latency_seconds" in c for c in plan.pass_criteria)
        assert any("Spec decoding désactivé" in n for n in plan.notes)

    def test_plan_without_latency_model_still_lists_steps(self):
        from vllm_configurator.catalog import load_catalog

        catalog = load_catalog()
        model = catalog.models["deepseek-v4.1-flash-1017"]
        catalog.models[model.id] = model.model_copy(update={"performance": None})
        try:
            plan = generate(req(workload={"target_concurrency": 600}), catalog).analysis.test_plan
        finally:
            catalog.models[model.id] = model
        assert plan.steps and all(s.itl_ms is None and s.expectation == "unknown" for s in plan.steps)
        assert "la mesure tranche" in plan.summary

    def test_step_budget_is_raised_when_its_prefill_load_needs_it(self):
        a = analysis(workload={"target_concurrency": 1000, "osl": {"p50": 250}}, slo={"itl_p95_ms": 100})
        big = [s for s in a.test_plan.steps if s.max_num_seqs >= 384]
        assert big and all(s.prefill_load_pct <= 80 for s in big)
        assert any(s.max_num_batched_tokens > 8192 for s in big)


def internals(**overrides):
    """(env, base point, state) of the analysis, for white-box checks."""
    from vllm_configurator import analysis as A
    from vllm_configurator.catalog import default_catalog
    from vllm_configurator.engine import _run_pipeline, build_context

    ctx = build_context(req(**overrides), default_catalog())
    state = _run_pipeline(ctx, ctx.default("gpu_memory_utilization")).state
    model, _ = A.build_perf_model(ctx, state)
    env, _ = A.build_env(ctx, state, state.get("kv_tokens"), model)
    base = A.Point(state["max_num_seqs"], state["max_num_batched_tokens"], state["long_prefill_token_threshold"],
                   ctx.request.num_instances)
    return env, base, state


class TestStepModel:
    def test_prefill_tokens_lengthen_the_step(self):
        # Somme des opérateurs : le calcul du prefill ne se cache plus sous la lecture des poids.
        b = analysis(workload={"target_concurrency": 400}).step_breakdown
        assert b["Step chargé, ITL p95 (ms)"] > 1.2 * b["Step de decode pur (ms)"]

    def test_step_counts_drafter_allreduce_and_fixed_costs(self):
        env, _, _ = internals()
        step = perf.step_time(env.model, 200, 4, env.context_tokens, 2048, env.prefill_context)
        assert {"dense", "experts", "attention", "allreduce", "fixed", "drafter"} <= set(step.ops)
        assert step.seconds == pytest.approx(sum(step.ops.values()), rel=1e-9)
        assert step.seconds > max(step.memory_s, step.compute_s)
        no_spec, _, _ = internals(spec={"enabled": False})
        assert "drafter" not in perf.step_time(no_spec.model, 200, 1, env.context_tokens, 0).ops

    def test_decode_and_prefill_parts_scale_separately(self):
        env, _, _ = internals()
        m = env.model
        step = perf.step_time(m, 200, 4, env.context_tokens, 2048, env.prefill_context)
        slower_prefill = perf.step_time(m.calibrated(1.0, 2.0), 200, 4, env.context_tokens, 2048, env.prefill_context)
        assert slower_prefill.seconds == pytest.approx(step.seconds + (step.seconds - step.decode_s), rel=1e-9)
        assert slower_prefill.decode_s == pytest.approx(step.decode_s)

    def test_attention_cost_is_derived_for_every_catalog_model(self):
        from vllm_configurator import analysis as A
        from vllm_configurator.catalog import default_catalog
        from vllm_configurator.engine import build_context

        catalog = default_catalog()
        for model_id in catalog.models:
            ctx = build_context(req(model_id=model_id), catalog)
            full, _, _, note = A.attention_cost(ctx)
            assert full > 0 and note, model_id
        _, _, tokens, note = A.attention_cost(build_context(req(), catalog))
        assert tokens == 512 + 128  # DeepSeek V4.1 : index_topk + sliding_window de config.json
        assert "indexeur" in note


class TestPrefillDistribution:
    def test_ttft_p95_reflects_cold_requests(self):
        # À hit rate constant, sous 5 % de requêtes froides le p95 peut même baisser (le prefill non caché part dans la
        # queue, au-delà du p95) ; au-delà de 5 %, le p95 est un prefill complet.
        default = checks(analysis(workload={"target_concurrency": 400}))["ttft"].estimate
        cold = checks(analysis(workload={"target_concurrency": 400, "cold_request_fraction": 0.08}))["ttft"].estimate
        assert cold > 3 * default

    def test_default_cold_fraction_is_capped_by_the_hit_rate(self):
        env, _, _ = internals()
        assert 0 < env.cold_fraction < 0.05
        assert env.hit_effective == pytest.approx(env.hit, abs=2e-3)

    def test_inconsistent_cold_fraction_is_reported(self):
        a = analysis(workload={"cold_request_fraction": 0.3})
        assert any("hit effectif" in n for n in a.notes)

    def test_load_test_without_prefix_reuse(self):
        warm = checks(analysis(workload={"target_concurrency": 400}))["ttft"].estimate
        cold = analysis(workload={"target_concurrency": 400, "prefix_cache_hit_rate": 0.0})
        assert checks(cold)["prefill_load"].estimate > 30
        assert checks(cold)["ttft"].estimate > 5 * warm

    def test_prefill_wait_uses_the_variance_of_the_prompts(self):
        from vllm_configurator import analysis as A

        env, base, _ = internals(workload={"target_concurrency": 1000, "osl": {"p50": 1000}})
        ev = A.evaluate(env, base)
        assert 0 < ev.rho < 1 and ev.servers >= 2
        assert env.u_scv > 5  # requêtes froides : service très dispersé
        assert ev.prefill_wait_s > 0 and ev.ttft_s > ev.own_prefill_s

    def test_fast_pass_matches_the_checks(self):
        from vllm_configurator import analysis as A

        env, base, _ = internals(workload={"target_concurrency": 1000, "osl": {"p50": 250}},
                                 slo={"ttft_p95_ms": 5000, "itl_p95_ms": 300})
        for pt in A.candidate_points(env, base, base.instances)[::7]:
            ev = A.evaluate(env, pt)
            assert A.passes(env, ev) == (A.worst(A.checks_for(env, ev)) == "ok"), pt


class TestCalibration:
    def test_ttft_measurement_recalibrates_the_prefill_only(self):
        a = analysis(slo={"ttft_p95_ms": 2000}, observed_perf={"ttft_p95_ms": 250})
        assert a.calibration is None and a.calibration_prefill
        assert checks(a)["ttft"].estimate == pytest.approx(250, rel=0.02)
        assert a.uncertainty == (0.8, 2.5) and a.uncertainty_ttft == (0.85, 1.15)

    def test_itl_measurement_keeps_the_ttft_range_wide(self):
        a = analysis(slo={"itl_p95_ms": 50}, observed_perf={"itl_p95_ms": 45})
        assert a.uncertainty == (0.85, 1.15) and a.uncertainty_ttft == (0.8, 2.5)
        assert any("même facteur que le decode" in n for n in a.notes)

    def test_irreconcilable_measurements_are_explained(self):
        a = analysis(slo={"itl_p95_ms": 50, "ttft_p95_ms": 2000}, observed_perf={"itl_p95_ms": 45, "ttft_p95_ms": 1500})
        assert any("non conciliables" in n for n in a.notes)
        assert checks(a)["itl"].estimate == pytest.approx(45, rel=0.01)  # la latence de génération reste recalée
        assert a.calibration_prefill is None


def test_plan_protocol_warns_about_load_test_artefacts():
    plan = analysis(workload={"target_concurrency": 400}, slo={"ttft_p95_ms": 2000}).test_plan
    text = " ".join(plan.protocol)
    assert "OSL fixe" in text and "raisonnement" in text and "hit rate" in text
