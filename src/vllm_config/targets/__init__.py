"""Generation targets.

v1 ships a single target, `vllm`. A target turns a DeploymentRequest into its own
artefact. Downstream targets (v2: `llmd_epp`, the llm-d EndpointPickerConfig) must
consume `GenerationResult.facts` (DeploymentFacts: KV tokens per instance, block
size, max_num_seqs, number of instances...) instead of recomputing them, e.g.
lruCapacityPerServer = formulas.lru_capacity_per_server(facts.kv_tokens_per_instance, facts.block_size).
"""

from __future__ import annotations

from typing import Callable

from ..domain import DeploymentRequest, GenerationResult
from ..engine import generate

Target = Callable[[DeploymentRequest], GenerationResult]

TARGETS: dict[str, Target] = {
    "vllm": generate,
}
