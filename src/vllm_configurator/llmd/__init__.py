"""llm-d target: EndpointPickerConfig (smart routing) derived from the vLLM configuration."""

from .render import render_epp_config, render_vllm_args
from .rules import generate_routing

__all__ = ["generate_routing", "render_epp_config", "render_vllm_args"]
