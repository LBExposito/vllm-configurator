"""working-set traces: workload read off recorded sessions, sessions in the test plan."""

from __future__ import annotations

import json

import pytest

from vllm_configurator import traces
from vllm_configurator.engine import generate
from vllm_configurator.scenario import dump_scenario, load_scenario, request_to_form, trace_to_form

from test_analysis import req


def request(prompt, output, gap=None, kind="tool", service=None):
    r = {"prompt_tokens": prompt, "output_tokens": output, "gap_s": gap, "after": "start" if gap is None else kind,
         "kind": "start" if gap is None else kind}
    if service is not None:
        r["service_s"] = service
    return r


def trace(*sessions, version=2):
    return {"format": traces.FORMAT, "version": version, "source": "claude",
            "sessions": [{"requests": list(s)} for s in sessions]}


# Deux sessions qui démarrent au-delà de 100k tokens ; la seconde est compactée (prompt qui rétrécit).
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
        # Froides : 2 premiers tours + 1 compaction ; chaudes : croissance au-delà du prompt et de la réponse précédents.
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
        # L'attente « idle » (2 000 s) est remplacée par la moyenne des attentes humaines (4 s), comme ws replay.
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
    assert set(form) <= set(request_to_form(req()))  # mêmes clés que le formulaire
    assert form["isl_p50"] > 100_000 and form["target_concurrency"] + form["paused_sessions"] == 100
    assert form["usage_type"] == "agentic" and form["warm_p95"] >= form["warm_p50"]
