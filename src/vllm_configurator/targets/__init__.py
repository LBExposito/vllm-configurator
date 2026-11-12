"""Generation targets.

A target turns a DeploymentRequest into its own artefact:

- `vllm`: the `vllm serve` configuration (GenerationResult);
- `llmd_epp`: the llm-d EndpointPickerConfig (RoutingResult). It consumes `GenerationResult.facts`
  (DeploymentFacts: KV tokens per instance, block size, max_num_seqs, number of instances...) instead of
  recomputing them, e.g. lruCapacityPerServer = formulas.lru_capacity_per_server(kv_tokens, block_size).
"""

from __future__ import annotations

from typing import Callable

from ..domain import DeploymentRequest, GenerationResult, RoutingResult
from ..engine import generate
from ..llmd import generate_routing

Target = Callable[[DeploymentRequest], GenerationResult | RoutingResult]


def generate_llmd_epp(request: DeploymentRequest) -> RoutingResult:
    return generate_routing(generate(request))


TARGETS: dict[str, Target] = {
    "vllm": generate,
    "llmd_epp": generate_llmd_epp,
}
