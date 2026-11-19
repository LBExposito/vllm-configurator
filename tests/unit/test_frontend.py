"""Python frontend of `vllm serve`: CPU model, sizing of --api-server-count / --renderer-num-workers, effect on SLOs."""

from __future__ import annotations

import math

import pytest

from vllm_configurator import frontend as fe
from vllm_configurator import render
from vllm_configurator.engine import generate

from test_analysis import checks, req

COSTS = fe.FrontendCosts(network_s=0.01, request_loop_s=1.5e-3, parse_s_per_token=0.05e-6,
                         template_s_per_token=0.15e-6, tokenize_s_per_token=0.5e-6, chunk_s=60e-6, utilization_max=0.7)

AGENTIC_SHORT = {"workload": {"target_concurrency": 400, "osl": {"p50": 250}}}
CHAT_NO_SPEC = {"workload": {"usage_type": "chat", "target_concurrency": 1000, "isl": {"p50": 3000, "p95": 6000},
                             "osl": {"p50": 300}}, "spec": {"enabled": False}}


def decisions(**overrides):
    result = generate(req(**overrides))
    return result, result.decision("api_server_count"), result.decision("renderer_num_workers")


class TestModel:
    def test_gil_load_adds_requests_template_and_outputs(self):
        load = fe.frontend_load(COSTS, 10, 5000, 30000, 30000**2, 1, 1)
        expected = 10 * (1.5e-3 + 0.05e-6 * 30000) + 10 * 0.15e-6 * 30000 + 5000 * 60e-6
        assert load.gil_utilization == pytest.approx(expected)
        assert load.gil_share["sorties"] == pytest.approx(0.3 / expected)  # COSTS : tokenize_gil_share = 0
        assert load.pool_utilization == pytest.approx(10 * 0.65e-6 * 30000)
        assert load.loop_wait_p95_s > load.loop_wait_s > 0

    def test_processes_split_the_load_threads_only_the_pool(self):
        one = fe.frontend_load(COSTS, 60, 5000, 30000, 30000**2, 1, 1)
        two = fe.frontend_load(COSTS, 60, 5000, 30000, 30000**2, 2, 1)
        threads = fe.frontend_load(COSTS, 60, 5000, 30000, 30000**2, 1, 4)
        assert two.gil_utilization == pytest.approx(one.gil_utilization / 2)
        assert threads.gil_utilization == pytest.approx(one.gil_utilization)  # le chat template tient le GIL
        assert threads.pool_utilization == pytest.approx(one.pool_utilization / 4)

    def test_saturation_makes_the_waits_infinite(self):
        load = fe.frontend_load(COSTS, 10, 20000, 3000, 3000**2, 1, 1)
        assert load.saturated and math.isinf(load.loop_wait_s)

    def test_sizing_follows_the_bottleneck(self):
        assert fe.size_frontend(COSTS, 5, 3000, 30000) == (1, 1)
        assert fe.size_frontend(COSTS, 20, 15000, 3000)[0] >= 2  # sorties : des processus, pas des threads
        servers, workers = fe.size_frontend(COSTS, 60, 1000, 30000)
        assert servers * workers >= 2  # longs prompts à fort débit : le rendu sature
        assert fe.size_frontend(COSTS, 20, 15000, 3000, max_servers=1)[0] == 1


class TestRule:
    def test_defaults_kept_when_they_hold(self):
        _, servers, workers = decisions(workload={"target_concurrency": 400})
        assert servers.value == 1 and not servers.emitted
        assert workers.value == 1 and not workers.emitted
        assert "suffit" in servers.justification

    def test_many_streams_need_more_api_servers(self):
        result, servers, workers = decisions(**CHAT_NO_SPEC)
        assert servers.value >= 2 and servers.emitted and workers.value == 1
        args = render.argv(result)
        assert args[args.index("--api-server-count") + 1] == str(servers.value)
        assert any("multiprocess" in w.message for w in result.warnings)

    def test_long_prompts_at_high_rate_size_the_renderer(self):
        _, servers, workers = decisions(**AGENTIC_SHORT)
        assert servers.value * workers.value >= 2
        assert "tokenisation" in workers.justification

    def test_rust_frontend_is_not_modelled(self):
        result, servers, workers = decisions(**CHAT_NO_SPEC, extra_env={"VLLM_USE_RUST_FRONTEND": "1"})
        assert not servers.emitted and not workers.emitted and "Rust" in servers.justification
        assert checks(result.analysis)["frontend"].status == "unknown"

    def test_runtime_lora_updates_cap_to_one_process(self):
        result, servers, _ = decisions(**CHAT_NO_SPEC, extra_env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"})
        assert servers.value == 1
        assert any("VLLM_ALLOW_RUNTIME_LORA_UPDATING" in w.message for w in result.warnings)

    def test_cpu_limit_caps_the_processes(self):
        result, servers, _ = decisions(**CHAT_NO_SPEC, cpu_cores_per_instance=6)
        assert servers.value == 1 and "plafonné" in servers.justification
        assert any(w.rule_id == "R-FE-03" for w in result.warnings)
        cpu = {i.key: i for i in result.indicators}["frontend_cpu"]
        assert cpu.value > 5  # engine core + 4 workers GPU + frontal

    def test_forced_value_is_a_known_flag_and_wins(self):
        result = generate(req(**CHAT_NO_SPEC, extra_args={"--api-server-count": 1}))
        assert not any(w.rule_id == "R-EXTRA-01" and w.severity == "critical" for w in result.warnings)
        assert not result.decision("api_server_count").emitted  # remplacé par l'argument supplémentaire
        front = checks(result.analysis)["frontend"]
        assert front.status in ("at_risk", "ko")
        assert "frontend_cpu" in [b.key for b in result.analysis.bottlenecks]


class TestEffectOnSlo:
    def test_saturated_renderer_makes_ttft_infinite(self):
        forced = generate(req(**AGENTIC_SHORT, extra_args={"--api-server-count": 1, "--renderer-num-workers": 1}))
        sized = generate(req(**AGENTIC_SHORT))
        assert checks(forced.analysis)["frontend"].status == "ko"
        assert checks(sized.analysis)["frontend"].status == "ok"
        assert checks(sized.analysis)["ttft"].estimate < checks(forced.analysis)["ttft"].estimate

    def test_frontend_wait_is_part_of_the_itl(self):
        result = generate(req(**CHAT_NO_SPEC, extra_args={"--api-server-count": 1}))
        b = result.analysis.step_breakdown
        assert checks(result.analysis)["itl"].estimate > b["Step chargé, ITL p95 (ms)"]

    def test_search_resizes_the_frontend(self):
        result = generate(req(**CHAT_NO_SPEC, extra_args={"--api-server-count": 1}, slo={"itl_p95_ms": 500}))
        changes = [r.changes for r in result.analysis.recommendations if r.kind == "param"]
        assert any(c.get("--api-server-count", 1) >= 2 for c in changes)
