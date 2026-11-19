"""Review report: assumptions with their origin, and the exported Markdown / JSON."""

from __future__ import annotations

import json

from vllm_configurator import report, traces
from vllm_configurator.catalog import default_catalog
from vllm_configurator.engine import generate
from vllm_configurator.llmd import generate_routing, render_epp_config

from test_analysis import req
from test_traces import AGENTIC_TRACE, RUN, TWO_SESSIONS


def by_label(rows):
    return {r["label"]: r for r in rows}


def full_report(result, **kwargs):
    catalog = default_catalog()
    routing = generate_routing(result, catalog)
    return report.build_report(result, catalog, routing, render_epp_config(routing, catalog), **kwargs)


class TestHypotheses:
    def test_missing_inputs_are_assumed(self):
        rows = by_label(report.hypotheses(req().model_copy(), generate(req())))
        assert rows["Part de requêtes froides"]["origin"] == report.ASSUMED
        assert rows["Prefix cache hit rate"]["origin"] == report.ASSUMED
        assert rows["Longueur d'acceptation a"]["origin"] == report.ASSUMED
        assert rows["KV par instance (tokens)"]["origin"] == report.LOGS
        assert rows["Modèle de latence"]["value"] == "non calibré"

    def test_given_and_imported_values_keep_their_origin(self):
        request = req(**AGENTIC_TRACE)
        rows = by_label(report.hypotheses(request, generate(request), {"cold_fraction": report.TRACE}))
        assert rows["Part de requêtes froides"]["origin"] == report.TRACE
        assert rows["Prefix cache hit rate"]["origin"] == report.ENTERED
        assert "Attente entre deux requêtes d'une session (s)" in rows

    def test_calibrated_model_is_reported(self):
        request = req(observed_perf={"itl_p95_ms": 30.0})
        rows = by_label(report.hypotheses(request, generate(request), {"measurements": report.REPLAY}))
        assert rows["Modèle de latence"]["value"] == "recalé sur la mesure"


class TestReport:
    def test_json_holds_everything_a_reviewer_needs(self):
        rep = full_report(generate(req()), name="ref")
        data = json.loads(report.report_json(rep))
        assert data["format"] == report.FORMAT and data["review"]["questions"]
        assert "--max-num-seqs" in data["configuration"]["vllm_args"]
        assert data["configuration"]["llmd_epp_config"].startswith("apiVersion:")
        assert data["analysis"]["model_inputs"]["cold_request_fraction"] > 0
        assert data["test_plan"]["steps"] and data["decisions"] and data["known_limits"]

    def test_markdown_asks_the_questions_and_lists_the_assumptions(self):
        md = report.render_markdown(full_report(generate(req())))
        assert "## Consignes de relecture" in md and "Verdict global" not in md
        assert "pertinent / à corriger / insuffisant" in md
        assert "| Part de requêtes froides |" in md and "supposée" in md
        assert "```bash" in md and "vllm serve" in md
        assert "Aucune mesure de latence fournie" in md

    def test_trace_and_replay_are_included(self):
        request = req(**AGENTIC_TRACE, observed_perf={"tpot_p95_ms": 20.0, "inflight_per_instance": 20})
        trace_rows = traces.summary_rows(traces.profile_trace(TWO_SESSIONS))
        rep = full_report(generate(request), trace_rows=trace_rows, run_profile=traces.profile_run(RUN))
        md = report.render_markdown(rep)
        assert "### Trace working-set importée" in md and "Rejeu working-set : valide 64 sessions" in md
        assert "### Capacité en sessions par instance" in md
        assert json.loads(report.report_json(rep))["measurements"]["replay"]["serves"] == 64
