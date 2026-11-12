"""Rules of the vLLM target. Each rule is a pure function (RuleContext, state) -> RuleOutput."""

from .context import RuleContext, RuleOutput
from .extra import decide_extra_args
from .diagnostics import check_support, decide_parallelism, itl_watch, resolve_workload, version_notes
from .kv_cache import (
    decide_block_size,
    decide_compilation,
    decide_gpu_memory_utilization,
    decide_kv_dtype,
    decide_kv_offload,
    decide_max_model_len,
    decide_prefix_caching,
    resolve_kv_capacity,
)
from .scheduler import (
    decide_batched_tokens,
    decide_chunked_prefill,
    decide_long_prefill,
    decide_max_num_seqs,
    indicate_scheduler,
)
from .spec_decode import decide_speculative

# Order matters: each rule reads the state produced by the previous ones.
PIPELINE = [
    check_support,
    resolve_workload,
    decide_parallelism,
    resolve_kv_capacity,
    decide_batched_tokens,
    decide_max_num_seqs,
    decide_long_prefill,
    decide_chunked_prefill,
    decide_prefix_caching,
    decide_kv_dtype,
    decide_speculative,
    decide_compilation,
    decide_gpu_memory_utilization,
    decide_block_size,
    decide_kv_offload,
    decide_max_model_len,
    indicate_scheduler,
    itl_watch,
    version_notes,
    decide_extra_args,
]

__all__ = ["PIPELINE", "RuleContext", "RuleOutput"]
