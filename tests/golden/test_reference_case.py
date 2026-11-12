"""Golden test: the reference case (§1) must produce the §7 configuration, or justified deviations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vllm_configurator import formulas as f
from vllm_configurator import render
from vllm_configurator.domain import DeploymentRequest
from vllm_configurator.engine import generate

CASE = Path(__file__).parent / "reference_case"


def _load(name: str):
    return yaml.safe_load((CASE / name).read_text(encoding="utf-8"))


def parse_args(args: list[str]) -> dict[str, object]:
    """--flag value pairs; bare flags map to True, --no-x to False; JSON values are decoded."""
    parsed: dict[str, object] = {}
    i = 0
    while i < len(args):
        flag = args[i]
        assert flag.startswith("--"), f"valeur orpheline : {flag}"
        nxt = args[i + 1] if i + 1 < len(args) else None
        if nxt is None or nxt.startswith("--"):
            if flag.startswith("--no-"):
                parsed["--" + flag[5:]] = False
            else:
                parsed[flag] = True
            i += 1
            continue
        parsed[flag] = json.loads(nxt) if nxt.startswith("{") else nxt
        i += 2
    return parsed


def diff(expected: dict, actual: dict) -> tuple[set, set]:
    flag_diffs, key_diffs = set(), set()
    for flag in expected.keys() | actual.keys():
        e, a = expected.get(flag), actual.get(flag)
        if isinstance(e, dict) and isinstance(a, dict):
            for key in e.keys() | a.keys():
                if key not in e:
                    key_diffs.add((flag, key, "added", json.dumps(a[key])))
                elif key not in a:
                    key_diffs.add((flag, key, "removed", json.dumps(e[key])))
                elif e[key] != a[key]:
                    key_diffs.add((flag, key, "changed", json.dumps(a[key])))
        elif flag not in expected:
            flag_diffs.add((flag, "added", json.dumps(a)))
        elif flag not in actual:
            flag_diffs.add((flag, "removed", json.dumps(e)))
        elif e != a:
            flag_diffs.add((flag, "changed", json.dumps(a)))
    return flag_diffs, key_diffs


@pytest.fixture(scope="module")
def result():
    return generate(DeploymentRequest(**_load("input.yaml")))


def test_config_matches_section_7_or_documented_deviations(result):
    expected = parse_args(_load("expected.yaml")["args"])
    actual = parse_args(render.argv(result))
    flag_diffs, key_diffs = diff(expected, actual)

    deviations = _load("deviations.yaml")
    for entry in deviations["flags"] + deviations["json_keys"]:
        assert entry["justification"].strip(), f"écart sans justification : {entry}"
    declared_flags = {(d["flag"], d["kind"], json.dumps(d["value"])) for d in deviations["flags"]}
    declared_keys = {(d["flag"], d["key"], d["kind"], json.dumps(d["value"])) for d in deviations["json_keys"]}

    assert flag_diffs - declared_flags == set(), "écarts de flags non justifiés dans deviations.yaml"
    assert declared_flags - flag_diffs == set(), "écarts déclarés qui ne se produisent plus : mettre à jour deviations.yaml"
    assert key_diffs - declared_keys == set(), "écarts de clés JSON non justifiés dans deviations.yaml"
    assert declared_keys - key_diffs == set(), "écarts de clés JSON déclarés qui ne se produisent plus"


def test_no_blocking_error(result):
    assert result.blocking_errors == []


def test_rationale_change_is_reflected_in_indicators(result):
    indicators = {i.key: i for i in result.indicators}
    assert indicators["decode_tokens_per_seq"].value == 4
    assert indicators["required_ratio"].value == pytest.approx(4.0)
    assert f.required_ratio(16384, 512, 4, 3.0) == pytest.approx(9.3, abs=0.05)
    assert indicators["reference_ratio"].value == 12.5


def test_reference_indicators(result):
    indicators = {i.key: i.value for i in result.indicators}
    assert indicators["kv_concurrency_p50"] == pytest.approx(3_600, rel=0.01)
    assert indicators["kv_concurrency_p95"] == pytest.approx(1_150, rel=0.01)
    assert indicators["decode_tokens_per_step"] == 2048
    assert indicators["dspark_active_seq_cap"] == 2048


def test_expected_warnings(result):
    categories = {w.category for w in result.warnings}
    assert "osl_suspect" in categories  # OSL P50 de 4K ≈ 15× la médiane par appel (§6)
    assert "isl_tail" in categories  # P99 666K à qualifier (§6)
    assert "memory" in categories  # KV mesuré avec une gpu-memory-utilization différente
    assert "lossy" not in categories  # FLy retiré


def test_facts_for_v2(result):
    facts = result.facts
    assert facts.kv_tokens_per_instance == 86_225_834
    assert (facts.block_size, facts.block_size_source) == (128, "backend_forced")
    assert f.lru_capacity_per_server(facts.kv_tokens_per_instance, facts.block_size) == pytest.approx(673_000, rel=0.001)


def test_block_size_never_emitted(result):
    assert "--block-size" not in render.argv(result)
