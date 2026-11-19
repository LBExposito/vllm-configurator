"""working-set traces: workload read off recorded sessions, sessions in the test plan."""

from __future__ import annotations

import json

import pytest

from vllm_configurator import traces
from vllm_configurator.engine import generate
from vllm_configurator.scenario import dump_scenario, load_scenario, request_to_form, run_to_form, trace_to_form

from test_analysis import checks, req


def request(prompt, output, gap=None, kind="tool", service=None):
    r = {"prompt_tokens": prompt, "output_tokens": output, "gap_s": gap, "after": "start" if gap is None else kind,
         "kind": "start" if gap is None else kind}
    if service is not None:
        r["service_s"] = service
    return r


def trace(*sessions, version=2):
    return {"format": traces.FORMAT, "version": version, "source": "claude",
            "sessions": [{"requests": list(s)} for s in sessions]}


# Two sessions that start past 100k tokens; the second one is compacted (its prompt shrinks).
TWO_SESSIONS = trace(
    [request(100_000, 1_000, service=20.0), request(103_000, 500, 5.0, service=10.0),
     request(104_500, 500, 5.0, service=10.0)],
    [request(120_000, 1_000, service=20.0), request(122_000, 1_000, 10.0, service=10.0),
     request(50_000, 1_000, 2_000.0, kind="idle", service=10.0), request(52_000, 1_000, 4.0, kind="think",
                                                                         service=10.0)],
)


class TestProfile:
    def test_cold_and_warm_requests_follow_the_replay_accounting(self):
        prof = traces.profile_trace(TWO_SESSIONS, shared_prefix_tokens=10_000)
        assert prof.sessions == 2 and prof.requests == 7
        # Cold: 2 first turns + 1 compaction; warm: growth past the previous prompt and reply.
        assert prof.resets == 1 and prof.cold_fraction == pytest.approx(3 / 7)
        cold_u = 90_000 + 110_000 + 40_000
        warm_u = [2_000, 1_000, 1_000, 1_000]
        total = 100_000 + 103_000 + 104_500 + 120_000 + 122_000 + 50_000 + 52_000
        assert prof.hit_rate == pytest.approx(1 - (cold_u + sum(warm_u)) / total)
        assert prof.cold_share_of_prefill == pytest.approx(cold_u / (cold_u + sum(warm_u)))
        assert prof.first_prompt_min == 100_000

    def test_uncached_reply_is_prefilled_again(self):
        cached = traces.profile_trace(TWO_SESSIONS)
        dropped = traces.profile_trace(TWO_SESSIONS, reply_cached=False)
        assert dropped.hit_rate < cached.hit_rate
        assert dropped.warm_uncached["p50"] > cached.warm_uncached["p50"]

    def test_waits_replace_inactive_gaps_and_give_the_busy_share(self):
        prof = traces.profile_trace(TWO_SESSIONS)
        # The « idle » wait (2,000 s) is replaced by the mean human wait (4 s), as ws replay does.
        assert prof.mean_gap_s == pytest.approx((5 + 5 + 10 + 4 + 4) / 5)
        service = (20 + 10 + 10 + 20 + 10 + 10 + 10) / 7
        assert prof.busy_share == pytest.approx(service / (service + prof.mean_gap_s))

    def test_workload_updates_split_sessions_into_inflight_and_paused(self):
        prof = traces.profile_trace(TWO_SESSIONS)
        updates = prof.workload_updates(sessions=200)
        assert updates["target_concurrency"] + updates["paused_sessions"] == 200
        assert updates["target_concurrency"] == round(200 * prof.busy_share)
        assert updates["isl"]["p50"] <= updates["isl"]["p95"] <= updates["isl"]["p99"]
        assert "target_concurrency" not in prof.workload_updates()

    def test_version_1_has_no_busy_share(self):
        doc = trace([request(100_000, 500), request(101_000, 500, 3.0)], version=1)
        prof = traces.profile_trace(doc)
        assert prof.busy_share is None and "target_concurrency" not in prof.workload_updates(sessions=50)
        assert any("version 1" in n for n in prof.notes)

    def test_large_first_prompts_warn_about_the_ws_default_shared_prefix(self):
        prof = traces.profile_trace(TWO_SESSIONS, shared_prefix_tokens=8_000)
        assert any("--shared-prefix-tokens 8000" in n for n in prof.notes)

    @pytest.mark.parametrize("data, message", [
        ("not json", "JSON invalide"),
        (json.dumps({"format": "autre"}), "Pas une trace"),
        (json.dumps({"format": traces.FORMAT, "version": 9, "sessions": []}), "Version"),
        (json.dumps({"format": traces.FORMAT, "version": 2, "sessions": []}), "aucune requête"),
        (json.dumps(trace([{"prompt_tokens": 0, "output_tokens": 1}])), "prompt_tokens"),
    ])
    def test_invalid_traces_are_rejected(self, data, message):
        with pytest.raises(traces.TraceError, match=message):
            traces.profile_trace(data)

    def test_warm_sigma_is_bounded(self):
        assert traces.warm_sigma(1000, 1000) == traces.SIGMA_BOUNDS[0]
        assert traces.warm_sigma(1000, 10 ** 9) == traces.SIGMA_BOUNDS[1]
        assert traces.warm_sigma(1000, 8000) == pytest.approx(1.264, abs=1e-3)


AGENTIC_TRACE = {"workload": {
    "isl": {"p50": 150_000, "p95": 220_000, "p99": 240_000}, "osl": {"p50": 800, "p95": 1_500},
    "prefix_cache_hit_rate": 0.96, "cold_request_fraction": 0.04, "warm_uncached_tokens": {"p50": 1_000, "p95": 8_000},
    "session_gap_s": 6.0, "target_concurrency": 270, "paused_sessions": 130}}


class TestPlanInSessions:
    def test_steps_give_sessions_and_the_ws_replay_command(self):
        result = generate(req(**AGENTIC_TRACE, slo={"ttft_p95_ms": 3000, "itl_p95_ms": 50}))
        plan = result.analysis.test_plan
        for step in plan.steps:
            low, high = step.sessions_per_instance
            assert step.inflight_per_instance <= low <= high
        protocol = " ".join(plan.protocol)
        assert "ws replay trace.json" in protocol and "--shared-prefix-tokens 0" in protocol
        assert "--max-warm-ttft-s 3" in protocol and "--min-decode-tok-s" in protocol and "--api chat" in protocol
        assert "vllm bench serve" not in protocol

    def test_sessions_that_overflow_the_kv_put_the_step_at_risk(self):
        tight = generate(req(**AGENTIC_TRACE, observed_kv={"kv_cache_tokens": 3_000_000}))
        steps = tight.analysis.test_plan.steps
        assert any(s.sessions_kv_pct and s.sessions_kv_pct > 90 for s in steps)
        over = [s for s in steps if s.sessions_kv_pct and s.sessions_kv_pct > 90]
        assert all(s.expectation in ("at_risk", "ko") and "évincées" in s.expectation_reason for s in over)

    def test_without_waits_the_plan_stays_in_requests_in_flight(self):
        result = generate(req())
        plan = result.analysis.test_plan
        assert all(s.sessions_per_instance is None for s in plan.steps)
        assert "vllm bench serve" in " ".join(plan.protocol)

    def test_warm_dispersion_comes_from_the_percentiles(self):
        result = generate(req(**AGENTIC_TRACE))
        assert any("lue sur les percentiles fournis" in n for n in result.analysis.notes)


def test_scenario_keeps_the_trace_fields():
    request_ = req(**AGENTIC_TRACE)
    loaded, _ = load_scenario(dump_scenario(request_))
    assert loaded == request_
    form = request_to_form(loaded)
    assert form["warm_p95"] == 8_000 and form["session_gap"] == 6.0


def test_trace_fills_known_form_fields():
    form = trace_to_form(traces.profile_trace(TWO_SESSIONS), sessions=100)
    assert set(form) <= set(request_to_form(req()))  # same keys as the form
    assert form["isl_p50"] > 100_000 and form["target_concurrency"] + form["paused_sessions"] == 100
    assert form["usage_type"] == "agentic" and form["warm_p95"] >= form["warm_p50"]



# --- ws replay results ----------------------------------------------------------------------------------------------

def row(ttft, span=10.0, tokens=501, prompt=150_000, cached=140_000, warm=True, sent=100.0):
    return {"seat": 0, "turn": 0, "kind": "hit" if warm else "first", "uncached": prompt - cached, "prompt": prompt,
            "out_tokens": tokens, "sent": sent, "ttft": ttft, "service": ttft + span, "error": None, "done": True,
            "ptok": prompt, "ctok": tokens, "ended": sent + ttft + span, "warm": warm, "cold": not warm,
            "expected": cached, "cached": cached, "span": span}


def run_step(sessions, passed, ttft, inflight, cold_every=10):
    rows = [row(ttft * (3 if i % cold_every == 0 else 1), cached=20_000 if i % cold_every == 0 else 140_000,
                warm=i % cold_every != 0) for i in range(40)]
    rows.append(row(0.1, sent=500.0))  # sent after the window: ignored
    return {"sessions": sessions, "passed": passed, "no_evidence": False, "window": [60.0, 360.0], "drain_s": 5.0,
            "streams_mean": inflight, "rows": rows, "reasons": [] if passed else ["warm time to first token: p90 9 s"],
            "checks": {"evictions": {"value": 0.0}},
            "server": {"prefix_hit_rate": 0.9, "kv_max": 0.4, "preemptions": 0, "waiting_max": 0}}


RUN = {"format": traces.RUN_FORMAT, "version": 1, "complete": True,
       "plan": {"shared_prefix_tokens": 20_000, "kv_block_tokens": 128, "routing": "sticky"},
       "endpoints": [{"base_url": "a"}, {"base_url": "b"}],
       "steps": [run_step(64, True, 1.0, 40.0), run_step(16, True, 0.5, 10.0), run_step(128, False, 6.0, 90.0)],
       "answer": {"serves": 64, "fails_at": 128, "capped": False}}


class TestRun:
    def test_steps_are_read_from_the_window(self):
        prof = traces.profile_run(json.dumps(RUN))
        assert [s.sessions for s in prof.steps] == [16, 64, 128]
        step = prof.steps[1]
        assert step.requests == 40 and step.inflight == 40.0
        assert step.tpot_p95_ms == pytest.approx(10.0 / 500 * 1e3)
        assert step.ttft_p95_ms == pytest.approx(3000.0)  # cold requests make the p95
        assert step.cold_fraction == pytest.approx(4 / 40)
        assert step.hit_rate == pytest.approx((36 * 140_000 + 4 * 20_000) / (40 * 150_000))
        assert prof.serves == 64 and prof.endpoints == 2

    def test_default_step_is_the_largest_pass(self):
        prof = traces.profile_run(RUN)
        assert prof.steps[prof.default_step()].sessions == 64
        failing = {**RUN, "steps": [run_step(32, False, 9.0, 30.0), run_step(8, False, 9.0, 8.0)]}
        prof = traces.profile_run(failing)
        assert prof.steps[prof.default_step()].sessions == 8

    def test_updates_bring_the_load_to_one_instance(self):
        prof = traces.profile_run(RUN)
        up = prof.updates(prof.default_step(), instances=2)
        assert up["observed_perf"]["inflight_per_instance"] == 20.0
        assert up["observed_perf"]["sessions_per_instance"] == 32.0
        assert up["observed_perf"]["capacity_sessions_per_instance"] == 32.0
        assert up["workload"]["cold_request_fraction"] == pytest.approx(0.1)

    def test_form_puts_the_tpot_where_the_target_is(self):
        prof = traces.profile_run(RUN)
        itl = run_to_form(prof, 1, 2, "itl")
        tpot = run_to_form(prof, 1, 2, "tpot")
        assert itl["tpot_measured"] == pytest.approx(20.0) and itl["itl_measured"] is None
        assert tpot["itl_measured"] == pytest.approx(20.0) and tpot["tpot_measured"] is None
        assert set(itl) <= set(request_to_form(req()))

    def test_invalid_results_are_rejected(self):
        with pytest.raises(traces.TraceError, match="ws replay"):
            traces.profile_run(json.dumps(TWO_SESSIONS))
        with pytest.raises(traces.TraceError, match="aucun palier"):
            traces.profile_run({**RUN, "steps": []})


class TestRecalibrationOnAReplay:
    def test_calibration_runs_at_the_measured_load(self):
        result = generate(req(**AGENTIC_TRACE, slo={"ttft_p95_ms": 3000, "itl_p95_ms": 50},
                              observed_perf={"tpot_p95_ms": 20.0, "inflight_per_instance": 40}))
        a = result.analysis
        assert a.calibration is not None
        assert any("Mesure faite à 40 requêtes en vol" in n for n in a.notes)
        assert any("reste extrapolée" in n for n in a.notes)  # ITL target, TPOT measurement

    def test_a_measurement_at_another_load_does_not_replace_the_estimate(self):
        result = generate(req(**AGENTIC_TRACE, slo={"ttft_p95_ms": 3000},
                              observed_perf={"ttft_p95_ms": 900.0, "inflight_per_instance": 3}))
        assert checks(result.analysis)["ttft"].measured is None


class TestSessionCapacity:
    def test_capacity_in_sessions_and_the_best_setting(self):
        result = generate(req(**AGENTIC_TRACE, slo={"ttft_p95_ms": 3000, "itl_p95_ms": 80}))
        cap = result.analysis.session_capacity
        assert cap.best_sessions >= cap.current_sessions > 0
        assert cap.best_sessions >= cap.best_inflight
        assert bool(cap.changes) == (cap.best_sessions > cap.current_sessions)
        assert "sessions par instance" in cap.summary

    def test_kv_of_waiting_sessions_can_be_the_limit(self):
        result = generate(req(**{**AGENTIC_TRACE, "workload": {**AGENTIC_TRACE["workload"], "session_gap_s": 600.0}},
                              observed_kv={"kv_cache_tokens": 3_000_000}))
        cap = result.analysis.session_capacity
        assert "KV des sessions" in cap.best_binding and "--kv-offloading-size" in cap.summary

    def test_measured_capacity_is_compared(self):
        result = generate(req(**AGENTIC_TRACE, observed_perf={"capacity_sessions_per_instance": 500.0}))
        cap = result.analysis.session_capacity
        assert cap.measured_sessions == 500.0 and "working-set a validé 500" in cap.summary
        assert any("Écart" in n for n in cap.notes)

    def test_no_capacity_without_waits(self):
        assert generate(req()).analysis.session_capacity is None
