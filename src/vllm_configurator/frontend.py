"""CPU model of the Python frontend of `vllm serve` (API server processes), used to size --api-server-count and
--renderer-num-workers and to add the frontend delays to TTFT and ITL.

What runs where (vLLM 0.31–0.33, Python frontend, verified in the source; see the frontend.* behaviors of the
version catalog):
- each API server process has one asyncio event loop and one GIL. The loop parses the HTTP request and the JSON of
  the messages, then streams the outputs: detokenization (tokenizers DecodeStream), deltas, reasoning and tool-call
  parsers, SSE, once per running sequence and per step (`output_handler` of AsyncLLM);
- the renderer pool (--renderer-num-workers threads per process) runs the chat template (HF and DeepSeek V4
  renderers) and the tokenization of the prompt. The chat template is Python (Jinja, or encode_messages for
  DeepSeek V4/V4.1) and holds the GIL: threads do not parallelize it, it competes with the loop. Tokenization goes
  through transformers `_encode_plus` → `tokenizers` `encode_batch`, whose Rust core runs without the GIL
  (`py.allow_threads` / `py.detach`); only the conversion of the result to Python lists keeps it
  (`tokenize_gil_share`);
- --api-server-count processes share the port and the requests; there is still one engine core.

So per process: GIL load = loop work + chat template; pool load = chat template + tokenization over the threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import perf as p


@dataclass(frozen=True)
class FrontendCosts:
    network_s: float                # hors CPU du pod : réseau, passage par l'EPP (TTFT seulement)
    request_loop_s: float           # boucle : HTTP, validation, routage, métriques, par requête
    parse_s_per_token: float        # boucle : JSON de l'historique de messages, par token de prompt
    template_s_per_token: float     # pool, tient le GIL : chat template
    tokenize_s_per_token: float     # pool : tokenisation (cœur Rust hors GIL)
    chunk_s: float                  # boucle : une sortie d'une séquence pour un step
    utilization_max: float
    tokenize_gil_share: float = 0.0  # part de la tokenisation qui garde le GIL (conversion du résultat en Python)

    @classmethod
    def from_heuristic(cls, value: dict) -> "FrontendCosts":
        return cls(
            network_s=value["network_ms"] * 1e-3, request_loop_s=value["request_loop_ms"] * 1e-3,
            parse_s_per_token=value["parse_loop_ms_per_1k_tokens"] * 1e-6,
            template_s_per_token=value["template_ms_per_1k_tokens"] * 1e-6,
            tokenize_s_per_token=value["tokenize_ms_per_1k_tokens"] * 1e-6,
            chunk_s=value["output_chunk_us"] * 1e-6, utilization_max=value["utilization_max"],
            tokenize_gil_share=value.get("tokenize_gil_share", 0.0),
        )

    @property
    def gil_render_s_per_token(self) -> float:
        """Part of the rendering, per prompt token, that holds the GIL: chat template + Python side of tokenization."""
        return self.template_s_per_token + self.tokenize_gil_share * self.tokenize_s_per_token

    def render_s(self, isl: float) -> float:
        """Service time of one request on the frontend, without waiting: loop work, chat template, tokenization."""
        return self.request_loop_s + (self.parse_s_per_token + self.template_s_per_token + self.tokenize_s_per_token) * isl


@dataclass(frozen=True)
class FrontendLoad:
    requests_per_s: float           # par instance
    chunks_per_s: float             # sorties streamées par seconde, par instance
    api_servers: int
    workers: int
    gil_utilization: float          # par processus API
    pool_utilization: float         # par processus API, sur ses threads
    loop_wait_s: float              # attente moyenne d'une tâche sur le GIL
    loop_wait_p95_s: float
    pool_wait_s: float              # attente moyenne d'un rendu de prompt dans le pool
    cores: float                    # cœurs CPU consommés par le frontal de l'instance
    gil_share: dict[str, float]     # part de la charge du GIL : requêtes, chat template, sorties

    @property
    def utilization(self) -> float:
        return max(self.gil_utilization, self.pool_utilization)

    @property
    def saturated(self) -> bool:
        return self.gil_utilization >= 1 or self.pool_utilization >= 1


def _moments(a: float, b: float, mean_x: float, m2_x: float) -> tuple[float, float]:
    """E[a + bX] and E[(a + bX)²]."""
    return a + b * mean_x, a * a + 2 * a * b * mean_x + b * b * m2_x


def frontend_load(costs: FrontendCosts, requests_per_s: float, chunks_per_s: float, isl_mean: float, isl_m2: float,
                  api_servers: int, workers: int) -> FrontendLoad:
    """Load of the frontend of one instance: requests spread evenly over the API server processes."""
    n, w = max(api_servers, 1), max(workers, 1)
    lam, chunks = requests_per_s / n, chunks_per_s / n
    req_mean, req_m2 = _moments(costs.request_loop_s, costs.parse_s_per_token, isl_mean, isl_m2)
    tpl_mean, tpl_m2 = _moments(0.0, costs.gil_render_s_per_token, isl_mean, isl_m2)
    gil_parts = {"requêtes": lam * req_mean, "rendu du prompt": lam * tpl_mean, "sorties": chunks * costs.chunk_s}
    gil = sum(gil_parts.values())
    if gil >= 1:
        loop_wait = loop_p95 = math.inf
    else:
        # M/G/1 sur le GIL (Pollaczek–Khinchine) : toutes les tâches, courtes (sorties) et longues (requêtes).
        second = lam * req_m2 + lam * tpl_m2 + chunks * costs.chunk_s ** 2
        loop_wait = second / (2 * (1 - gil))
        loop_p95 = loop_wait / gil * math.log(gil / 0.05) if gil > 0.05 else 0.0
    pool_mean, pool_m2 = _moments(0.0, costs.template_s_per_token + costs.tokenize_s_per_token, isl_mean, isl_m2)
    pool_rho = lam * pool_mean / w
    scv = max(pool_m2 / pool_mean**2 - 1, 0.0) if pool_mean > 0 else 0.0
    p_wait, scale = p.mgc_wait(pool_rho, w, pool_mean, scv)
    pool_wait = math.inf if math.isinf(scale) else p_wait * scale
    cores = (requests_per_s * (req_mean + (costs.template_s_per_token + costs.tokenize_s_per_token) * isl_mean)
             + chunks_per_s * costs.chunk_s)
    total = gil or 1.0
    return FrontendLoad(
        requests_per_s=requests_per_s, chunks_per_s=chunks_per_s, api_servers=n, workers=w, gil_utilization=gil,
        pool_utilization=pool_rho, loop_wait_s=loop_wait, loop_wait_p95_s=loop_p95, pool_wait_s=pool_wait, cores=cores,
        gil_share={key: value / total for key, value in gil_parts.items()},
    )


def size_frontend(costs: FrontendCosts, requests_per_s: float, chunks_per_s: float, isl_mean: float,
                  max_servers: int = 64) -> tuple[int, int]:
    """Smallest (api servers, renderer workers) keeping the GIL and the pool of every process under utilization_max."""
    umax = costs.utilization_max
    req_mean = costs.request_loop_s + costs.parse_s_per_token * isl_mean
    gil_total = requests_per_s * (req_mean + costs.gil_render_s_per_token * isl_mean) + chunks_per_s * costs.chunk_s
    servers = min(max(1, math.ceil(gil_total / umax)), max_servers)
    pool_total = requests_per_s / servers * (costs.template_s_per_token + costs.tokenize_s_per_token) * isl_mean
    workers = min(max(1, math.ceil(pool_total / umax)), 64)
    return servers, workers
