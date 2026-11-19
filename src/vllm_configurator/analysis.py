"""SLO analysis: will the generated configuration meet the targets, what is the bottleneck, what to change.

The analysis evaluates an operating point (max_num_seqs, token budget, long prefill threshold,
number of instances) with two kinds of checks:

- structural checks, which need no hardware model: KV occupancy, prefill load of the budget,
  queueing for a sequence slot;
- latency checks (ITL or TPOT p95, TTFT p95), from the step model of perf.py, recalibrated by
  the measured ITL (decode part) and TTFT (prefill part) when provided.

Percentiles are computed as percentiles, not as sums of means:
- the tokens to prefill follow a distribution (cold requests prefilled in full, warm requests with a
  heavy-tailed uncached part), not ISL P95 × (1 − mean hit rate);
- the wait for the prefill comes from an M/G/c queue (variance of that distribution, several prefills
  in parallel within the budget), and TTFT p95 is the 95th percentile of frontend + slot wait + prefill
  wait + own prefill;
- TPOT p95 divides the mean step by the acceptance length of the 5 % least accepted requests.

It then classifies the bottleneck and searches the parameter grid (and then the number of
instances) for an operating point that meets every target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

from . import formulas as f
from . import frontend as fe
from . import perf as p
from .catalog.loader import version_key
from .domain import Bottleneck, CheckStatus, Recommendation, SloAnalysis, SloCheck, TestPlan, TestStep
from .rules.context import KB, RuleContext, fmt_int

GIB = 1024**3
RANK: dict[CheckStatus, int] = {"unknown": -1, "ok": 0, "at_risk": 1, "ko": 2}
BUDGET_GRID = [4096, 8192, 12288, 16384, 24576, 32768]


@dataclass(frozen=True)
class Point:
    """An operating point of the deployment."""

    max_num_seqs: int
    budget: int
    threshold: int
    instances: int


@dataclass(frozen=True)
class Env:
    """Everything that does not change between operating points."""

    ctx: RuleContext
    k: int
    d: int
    a: float
    hit: float
    kv_tokens: int | None
    model: p.PerfModel | None
    dspark_cap: bool
    # Tokens à prefiller par requête : (probabilité, tokens non cachés U, ISL), triés par U.
    dist: tuple[tuple[float, float, float], ...] = ((1.0, 0.0, 0.0),)
    hit_effective: float = 0.0
    cold_fraction: float = 0.0
    # Longueur d'acceptation des 5 % de requêtes les moins acceptées (TPOT p95).
    a_low: float = 1.0
    # Frontal Python : coûts CPU (None : non modélisé, frontend Rust), processus API et threads du renderer configurés.
    # frontend_autosize : redimensionnés point par point (recherche de réglages : le CPU suit la charge).
    frontend_costs: fe.FrontendCosts | None = None
    api_servers: int = 1
    renderer_workers: int = 1
    frontend_autosize: bool = False
    max_api_servers: int = 64
    # Incertitude multiplicative (bas, haut) sur la partie decode et sur la partie prefill des durées.
    spread: tuple[float, float] = (1.0, 1.0)
    spread_prefill: tuple[float, float] = (1.0, 1.0)

    @property
    def wl(self):
        return self.ctx.request.workload

    @property
    def isl_mean(self) -> float:
        return f.mean_from_percentiles(self.wl.isl.p50, self.wl.isl.p95, self.wl.isl.p99)

    @property
    def osl_mean(self) -> float:
        return f.mean_from_percentiles(self.wl.osl.p50, self.wl.osl.p95, self.wl.osl.p99)

    @property
    def context_tokens(self) -> float:
        # Contexte moyen d'une séquence en decode : prompt moyen + moitié de la sortie moyenne.
        return self.isl_mean + self.osl_mean / 2

    @cached_property
    def isl_eff_mean(self) -> float:
        """Mean tokens to prefill per request (E[U])."""
        return sum(prob * u for prob, u, _ in self.dist)

    @cached_property
    def u_scv(self) -> float:
        """Squared coefficient of variation of U (variance / mean²)."""
        mean = self.isl_eff_mean
        if mean <= 0:
            return 0.0
        return max(sum(prob * u * u for prob, u, _ in self.dist) / mean**2 - 1.0, 0.0)

    @cached_property
    def prefill_context(self) -> float:
        """Mean position of a prefilled token in its prompt (token-weighted): what its attention reads."""
        weight = sum(prob * u for prob, u, _ in self.dist)
        if weight <= 0:
            return 0.0
        return sum(prob * u * (isl - u / 2) for prob, u, isl in self.dist) / weight

    @property
    def metric(self) -> str:
        return self.ctx.request.slo.itl_metric

    @property
    def step_target_ms(self) -> float | None:
        """Generation-latency target expressed as a step duration (TPOT p95 × a of the slow requests)."""
        target = self.ctx.request.slo.itl_p95_ms
        if target is None:
            return None
        return target * self.a_low if self.metric == "tpot" else target

    @cached_property
    def isl_eff_p95(self) -> float:
        """95th percentile of the tokens to prefill per request."""
        return f.quantile_of_points(list(self.dist), 0.95)

    @property
    def isl_cold(self) -> int:
        return self.wl.isl.p95 or self.wl.isl.p50

    @cached_property
    def attn_correction_flops(self) -> tuple[float, ...]:
        """Per point of `dist`: attention FLOPs of its own prefill minus the same tokens at the mean prefill context
        (the chunk steps are priced at that mean context)."""
        m = self.model
        if m is None:
            return tuple(0.0 for _ in self.dist)
        mean = m.attention_flops_per_query(self.prefill_context)
        return tuple(u * (m.attention_flops_per_query(isl - u / 2) - mean) for _, u, isl in self.dist)

    @cached_property
    def isl_m2(self) -> float:
        """Second moment of the full prompt length (the frontend renders and tokenizes the whole prompt)."""
        return sum(prob * isl * isl for prob, _, isl in self.dist)

    def frontend_s(self, isl: float) -> float:
        """Time outside the GPU before the first token, without waiting: network, EPP, HTTP, chat template,
        tokenization of the prompt."""
        costs = self.frontend_costs
        return 0.0 if costs is None else costs.network_s + costs.render_s(isl)

    def frontend_at(self, requests_per_s: float, chunks_per_s: float) -> fe.FrontendLoad | None:
        """Load of the frontend for this traffic, with the configured (or re-sized) processes and threads."""
        costs = self.frontend_costs
        if costs is None:
            return None
        servers, workers = self.api_servers, self.renderer_workers
        if self.frontend_autosize:
            servers, workers = fe.size_frontend(costs, requests_per_s, chunks_per_s, self.isl_mean, self.max_api_servers)
        return fe.frontend_load(costs, requests_per_s, chunks_per_s, self.isl_mean, self.isl_m2, servers, workers)


@dataclass(frozen=True)
class Evaluation:
    point: Point
    per_instance: int | None        # concurrence cible par instance
    active: int                     # séquences en decode
    capacity: float                 # tokens de prefill disponibles par step
    rho: float                      # charge de prefill du budget
    kv_usage: float | None
    slot_wait_s: float
    decode: p.StepTime | None = None
    loaded: p.StepTime | None = None
    average: p.StepTime | None = None
    chunk: float = 0
    chunk_step: p.StepTime | None = None
    servers: int = 1                # prefills menés de front dans le budget
    p_wait: float = 0.0             # probabilité d'attendre avant son prefill
    prefill_wait_s: float = 0.0     # attente avant prefill, moyenne
    prefill_wait_p95_s: float = 0.0
    own_prefill_s: float = 0.0      # prefill d'un prompt au p95 des tokens non cachés
    frontend_s: float = 0.0         # frontal (tokenisation…) pour l'ISL P95, attentes comprises
    frontend: fe.FrontendLoad | None = None
    ttft_s: float | None = None     # TTFT p95 (quantile de la somme des composantes)
    ttft_cold_s: float | None = None
    tpot_s: float | None = None     # TPOT p95

    @property
    def itl_ms(self) -> float | None:
        """ITL p95: one loaded step, plus the p95 wait of a streamed output on the GIL of its API process."""
        if self.loaded is None:
            return None
        return self.loaded.ms + (self.frontend.loop_wait_p95_s * 1e3 if self.frontend else 0.0)

    @property
    def tpot_ms(self) -> float | None:
        return None if self.tpot_s is None else self.tpot_s * 1e3

    @property
    def ttft_ms(self) -> float | None:
        return None if self.ttft_s is None else self.ttft_s * 1e3

    @property
    def ttft_cold_ms(self) -> float | None:
        return None if self.ttft_cold_s is None else self.ttft_cold_s * 1e3

    def throughput(self, a: float) -> float | None:
        """Output tokens per second and per instance."""
        if self.average is None:
            return None
        return self.active * a / self.average.seconds


# --- Model construction ------------------------------------------------------

def kv_bytes_per_token_per_gpu(ctx: RuleContext, state: dict[str, Any]) -> tuple[float | None, str]:
    kv, tp = ctx.model.kv_cache, ctx.request.gpus_per_instance
    if kv.bytes_per_token is not None and kv.replicated_across_tp is not None:
        if kv.replicated_across_tp:
            return kv.bytes_per_token, f"{kv.bytes_per_token:g} o par token, répliqués sur chaque rang TP (catalogue)"
        shards = min(tp, ctx.model.architecture.num_key_value_heads or tp)
        return kv.bytes_per_token / shards, f"{kv.bytes_per_token:g} o par token répartis sur {shards} rangs (catalogue)"
    observed = ctx.request.observed_kv
    weights = ctx.model.weights.checkpoint_bytes
    util = (observed.gpu_memory_utilization if observed else None) or state.get("gpu_memory_utilization")
    if observed and weights and util:
        free = ctx.gpu.memory.hbm_gb * 1e9 * util - weights / tp
        if free > 0:
            value = free / observed.kv_cache_tokens
            return value, f"≈ {value:.0f} o par token et par GPU, déduits des logs (mémoire libre / tokens de KV, borne haute)"
    return None, "octets de KV par token inconnus : la lecture du KV en decode est ignorée (latence sous-estimée)"


def build_perf_model(ctx: RuleContext, state: dict[str, Any]) -> tuple[p.PerfModel | None, list[str]]:
    gpu, model, perf = ctx.gpu, ctx.model, ctx.model.performance
    missing = []
    if perf is None:
        missing.append(f"paramètres de performance du modèle (bloc performance de catalog/models/{model.id}.yaml)")
    if gpu.bandwidth is None or gpu.bandwidth.hbm_tb_s is None:
        missing.append(f"bande passante HBM du {gpu.display_name}")
    if perf is not None and (gpu.compute is None or perf.compute_dtype not in gpu.compute.dense_tflops):
        missing.append(f"débit crête {perf.compute_dtype if perf else ''} du {gpu.display_name} (bloc compute)")
    if missing:
        return None, ["Modèle de latence indisponible : " + " ; ".join(missing) + ". Seuls les contrôles de capacité sont faits."]

    h = lambda key: ctx.heuristic(key).value  # noqa: E731
    arch = model.architecture
    moe = arch.moe
    if moe is not None:
        per_expert = 3 * arch.hidden_size * perf.moe_intermediate_size * perf.moe_layers
        routed = per_expert * moe.n_routed_experts
        routed_active = per_expert * moe.num_experts_per_tok
        experts_per_tok, n_experts = moe.num_experts_per_tok, moe.n_routed_experts
        routed_bpp = perf.routed_expert_bytes_per_param
    else:
        routed = routed_active = 0.0
        experts_per_tok, n_experts, routed_bpp = 0, 0, 0.0
    dense = max(perf.active_params_per_output_token - routed_active, 0.0)
    kv_bpt, kv_note = kv_bytes_per_token_per_gpu(ctx, state)
    tp = ctx.request.gpus_per_instance
    attn_full, attn_sparse, attn_tokens, attn_note = attention_cost(ctx)
    nvlink = gpu.bandwidth.nvlink_gpu_to_gpu_tb_s
    allreduce = nvlink * 1e12 / 2 * h("perf_nvlink_efficiency") if nvlink and tp > 1 else None
    drafter, drafter_note = drafter_fraction(ctx, state)
    m = p.PerfModel(
        tp=tp,
        hbm_bytes_per_s=gpu.bandwidth.hbm_tb_s * 1e12 * h("perf_hbm_efficiency"),
        flops_per_s=gpu.compute.dense_tflops[perf.compute_dtype] * 1e12 * h("perf_mfu"),
        overhead_s=arch.num_hidden_layers * h("perf_overhead_per_layer_ms") * 1e-3,
        active_params_decode=perf.active_params_per_output_token,
        active_params_prefill=perf.active_params_per_prompt_token or perf.active_params_per_output_token,
        dense_params=dense, dense_bytes_per_param=perf.dense_bytes_per_param,
        routed_params=routed, routed_active_params=routed_active, routed_bytes_per_param=routed_bpp,
        experts_per_tok=experts_per_tok, n_experts=n_experts, kv_bytes_per_token_per_gpu=kv_bpt,
        attn_full_flops=attn_full, attn_sparse_flops=attn_sparse, attn_sparse_tokens=attn_tokens,
        hidden_size=arch.hidden_size, num_layers=arch.num_hidden_layers, allreduce_bytes_per_s=allreduce,
        mixed_overhead_factor=h("perf_mixed_step_overhead_factor"),
        per_seq_overhead_s=h("perf_overhead_per_seq_us") * 1e-6, drafter_fraction=drafter,
    )
    notes = [
        f"Modèle de latence : somme des opérateurs du step, chacun borné par son roofline (lecture mémoire ou calcul),"
        f" sur {gpu.display_name} à {gpu.bandwidth.hbm_tb_s:g} To/s × {h('perf_hbm_efficiency'):g} et"
        f" {fmt_int(gpu.compute.dense_tflops[perf.compute_dtype])} TFLOPS {perf.compute_dtype} × {h('perf_mfu'):g}."
        f" Surcoût fixe {m.overhead_s * 1e3:.1f} ms par step ({arch.num_hidden_layers} couches ×"
        f" {h('perf_overhead_per_layer_ms'):g} ms, × {m.mixed_overhead_factor:g} sur un step mixte prefill + decode),"
        f" plus {h('perf_overhead_per_seq_us'):g} µs par séquence."
        f" Poids lus par step : {dense / 1e9:.1f} G paramètres non routés"
        + (f" + la part des {routed / 1e9:.0f} G paramètres d'experts routés touchée par au moins un token." if routed else ".")
        + f" KV : {kv_note}. Attention : {attn_note}."
        + (f" All-reduce TP : 2 par couche sur NVLink à {nvlink / 2:g} To/s par sens × {h('perf_nvlink_efficiency'):g}."
           if allreduce else (" All-reduce TP : volume non modélisé (débit NVLink inconnu)." if tp > 1 else ""))
        + (f" Drafter : {drafter_note}." if drafter_note else ""),
    ]
    if ctx.model.kv_cache.bytes_per_token is not None and set(arch.attention) & SPARSE_ATTENTION:
        notes.append("Attention sparse : le modèle suppose la lecture de tout le contexte en decode (borne haute).")
    return m, notes


SPARSE_ATTENTION = {"sparse", "compressed_sparse", "sparse_dsa", "indexed_sparse", "sparse_qsa"}
PARALLEL_DRAFTERS = {"dspark", "dflash"}
FREE_DRAFTERS = {"ngram", "ngram_gpu", "suffix"}


def attention_cost(ctx: RuleContext) -> tuple[float, float, float, str]:
    """Attention FLOPs per query token: (per context token, per selected token, selected tokens, explanation).

    QKᵀ and PV cost 2 × heads × (d_qk + d_v) FLOPs per (query, key) pair and per softmax layer (linear attention
    layers have no context term). A sparse attention attends in full to top-k + sliding window tokens and runs an
    indexer over the whole context, at `perf_sparse_indexer_cost_ratio` of a full pair. Values set in the
    performance block of the model override the derivation."""
    model = ctx.model
    perf, arch, raw = model.performance, model.architecture, model.architecture.raw
    if perf is not None and perf.attention_flops_per_context_token is not None:
        return (perf.attention_flops_per_context_token, perf.attention_sparse_flops_per_token or 0.0,
                float(perf.attention_sparse_tokens or 0), "valeurs du bloc performance du catalogue")
    heads = arch.num_attention_heads
    fallback = arch.head_dim or arch.hidden_size // heads
    d_qk = raw.get("qk_head_dim") or ((raw.get("qk_nope_head_dim") or 0) + (raw.get("qk_rope_head_dim") or 0)) or fallback
    d_v = raw.get("v_head_dim") or fallback
    pair = 2.0 * heads * (d_qk + d_v)
    layers = raw.get("sparse_attention_layers") or raw.get("full_attention_layers") or arch.num_hidden_layers
    shape = (f"{layers} couches × 2 × {heads} têtes × ({d_qk} + {d_v}) = {pair * layers / 1e6:.1f} MFLOP par paire"
             " (requête, token de contexte), dérivé de config.json")
    if set(arch.attention) & SPARSE_ATTENTION:
        topk = raw.get("index_topk") or raw.get("indexer_budget")
        if topk:
            window = raw.get("sliding_window") or 0
            ratio = ctx.heuristic("perf_sparse_indexer_cost_ratio").value
            return (layers * pair * ratio, layers * pair, float(topk + window),
                    f"sparse, {shape} sur {topk + window} tokens sélectionnés + indexeur sur tout le contexte à"
                    f" {ratio:g} paire (🔶)")
        return layers * pair, 0.0, 0.0, f"sparse sans top-k connu, compté dense (borne haute) : {shape}"
    return layers * pair, 0.0, 0.0, f"dense, {shape}"


def drafter_fraction(ctx: RuleContext, state: dict[str, Any]) -> tuple[float, str | None]:
    """Cost of the drafter pass as a fraction of a target decode step."""
    k, method = state.get("num_speculative_tokens") or 0, state.get("spec_method")
    if not k or not method or method in FREE_DRAFTERS:
        return 0.0, None
    h = ctx.heuristic("perf_drafter_step_fraction").value
    if method in PARALLEL_DRAFTERS:
        return h["parallel"], f"{method}, une passe par step ≈ {h['parallel']:.0%} d'un step de decode (🔶)"
    value = h["autoregressive_per_token"] * k
    return value, f"{method}, {k} passes par step ≈ {value:.0%} d'un step de decode (🔶)"


# --- Evaluation of one operating point ---------------------------------------

def _load(env: Env, pt: Point, inflight: int | None) -> tuple[int | None, int, float, float, float, float | None]:
    """(requests per instance, active sequences, prefill capacity per step, arrivals per step, prefill load, KV usage)."""
    wl = env.wl
    if inflight is not None:
        per_instance = inflight
    else:
        per_instance = math.ceil(wl.target_concurrency / pt.instances) if wl.target_concurrency else None
    active = min(pt.max_num_seqs, per_instance) if per_instance else pt.max_num_seqs
    capacity = pt.budget - active * env.d
    # Régime stable : S·a/OSL requêtes finissent par step et sont remplacées (§3.1), longueurs moyennes.
    arrivals = active * env.a / env.osl_mean
    rho = math.inf if capacity <= 0 else arrivals * env.isl_eff_mean / capacity
    kv_usage = active * env.context_tokens / env.kv_tokens if env.kv_tokens else None
    return per_instance, active, capacity, arrivals, rho, kv_usage


def evaluate(env: Env, pt: Point, inflight: int | None = None) -> Evaluation:
    """Evaluate an operating point. `inflight` forces the in-flight requests per instance (test steps)."""
    per_instance, active, capacity, arrivals, rho, kv_usage = _load(env, pt, inflight)
    ev = Evaluation(point=pt, per_instance=per_instance, active=active, capacity=capacity, rho=rho,
                    kv_usage=kv_usage, slot_wait_s=0.0)
    m = env.model
    if m is None or capacity <= 0:
        return ev

    tokens_per_seq = env.k + 1
    ctx_tokens = env.context_tokens
    pf_ctx = env.prefill_context
    chunk = capacity if active <= 1 else min(pt.threshold, capacity)
    decode = p.step_time(m, active, tokens_per_seq, ctx_tokens, 0)
    average = p.step_time(m, active, tokens_per_seq, ctx_tokens, min(rho, 1.0) * capacity, pf_ctx)

    # Prefills en cours dans un step : Poisson de moyenne arrivées × steps de prefill par requête (loi de Little ;
    # chaque requête occupe au moins un step, les longues un par morceau), chacun apportant en moyenne τ tokens. L'ITL p95
    # est celle du step au 95e centile de ce nombre (0 si moins de 5 % des steps portent du prefill).
    steps_per_prefill = sum(prob * max(1, math.ceil(u / chunk)) for prob, u, _ in env.dist)
    in_progress = arrivals * steps_per_prefill
    n95 = _poisson_quantile(in_progress, 0.95)
    if n95:
        tau = min(chunk, env.isl_eff_mean / steps_per_prefill) if steps_per_prefill else chunk
        p95_prefill = min(capacity, max(min(env.isl_eff_p95, chunk), n95 * tau))
    else:
        p95_prefill = 0.0
    loaded = p.step_time(m, active, tokens_per_seq, ctx_tokens, p95_prefill, pf_ctx)

    chunk_step = p.step_time(m, active, tokens_per_seq, ctx_tokens, min(capacity, max(chunk, min(rho, 1.0) * capacity)),
                             pf_ctx)
    slot_wait = 0.0
    if per_instance and per_instance > pt.max_num_seqs:
        duration = env.osl_mean / env.a * average.seconds
        slot_wait = (per_instance - pt.max_num_seqs) / pt.max_num_seqs * duration

    # Attente avant prefill : M/G/c. Une requête en prefill occupe le budget pendant n = ⌈U / morceau⌉ steps, en y
    # prenant τ tokens en moyenne (une petite requête ne prend que ses tokens, le reste sert les suivantes) : le budget
    # mène de front c = capacité / τ prefills. Service = n steps, de variance celle de n (requêtes froides comprises).
    second = sum(prob * max(1, math.ceil(u / chunk)) ** 2 for prob, u, _ in env.dist)
    scv = max(second / steps_per_prefill**2 - 1.0, 0.0) if steps_per_prefill else 0.0
    tau = min(chunk, env.isl_eff_mean / steps_per_prefill) if steps_per_prefill else chunk
    servers = max(1, math.floor(capacity / max(tau, 1.0)))
    p_wait, wait_scale = p.mgc_wait(rho, servers, steps_per_prefill * average.seconds, scv)
    wait_p95 = wait_scale * math.log(p_wait / 0.05) if p_wait > 0.05 else 0.0

    cs = chunk_step.seconds

    def own_prefill(u: float, isl: float) -> float:
        # Morceaux au rythme d'un step chargé (contexte moyen du prefill), corrigés du contexte propre au prompt.
        correction = p.prefill_attention_seconds(m, u, isl - u / 2) - p.prefill_attention_seconds(m, u, pf_ctx)
        return max(max(1, math.ceil(u / chunk)) * cs + correction, cs)

    # Frontal : requêtes et sorties streamées par seconde au rythme du step moyen.
    front = env.frontend_at(arrivals / average.seconds, active / average.seconds)
    front_wait = (front.loop_wait_s + front.pool_wait_s) if front else 0.0

    # Une requête arrive en cours de step : elle attend en moyenne un demi-step avant d'être ordonnancée.
    fixed = slot_wait + 0.5 * average.seconds + front_wait
    per_flop = m.prefill_factor / (m.tp * m.flops_per_s)
    points = [(prob, fixed + env.frontend_s(isl) + max(max(1, math.ceil(u / chunk)) * cs + corr * per_flop, cs))
              for (prob, u, isl), corr in zip(env.dist, env.attn_correction_flops)]
    ttft = math.inf if math.isinf(front_wait) else p.quantile_of_sum(0.95, points, p_wait, wait_scale)
    u95 = env.isl_eff_p95
    isl95 = next((isl for prob, u, isl in env.dist if u >= u95), env.isl_cold)
    cold = (fixed + env.frontend_s(env.isl_cold) + own_prefill(env.isl_cold, env.isl_cold)
            + (p_wait * wait_scale if not math.isinf(wait_scale) else math.inf))
    return replace(ev, decode=decode, loaded=loaded, average=average, chunk=chunk, chunk_step=chunk_step,
                   slot_wait_s=slot_wait, servers=servers, p_wait=p_wait,
                   prefill_wait_s=p_wait * wait_scale if not math.isinf(wait_scale) else math.inf,
                   prefill_wait_p95_s=wait_p95, own_prefill_s=own_prefill(u95, isl95),
                   frontend_s=env.frontend_s(isl95) + front_wait, frontend=front, ttft_s=ttft, ttft_cold_s=cold,
                   tpot_s=math.inf if front and front.gil_utilization >= 1 else average.seconds / env.a_low)


def _poisson_quantile(mean: float, q: float) -> int:
    """Smallest n with P(N ≤ n) ≥ q for N ~ Poisson(mean)."""
    if mean <= 0:
        return 0
    if mean > 50:  # approximation normale (exp(−mean) sous-déborderait)
        return math.ceil(mean + f.normal_ppf(q) * math.sqrt(mean))
    term = cdf = math.exp(-mean)
    n = 0
    while cdf < q and n < 10_000:
        n += 1
        term *= mean / n
        cdf += term
    return n


def with_factor(env: Env, which: int) -> Env:
    """The environment at the optimistic (0) or pessimistic (1) end of the model uncertainty."""
    if env.model is None:
        return env
    decode, prefill = env.spread[which], env.spread_prefill[which]
    if decode == 1.0 and prefill == 1.0:
        return env
    return replace(env, model=env.model.scaled(decode, prefill))


def evaluate_range(env: Env, pt: Point, inflight: int | None = None) -> tuple[Evaluation, Evaluation, Evaluation]:
    """(central, optimistic, pessimistic) evaluations under the model uncertainty."""
    return (evaluate(env, pt, inflight), evaluate(with_factor(env, 0), pt, inflight),
            evaluate(with_factor(env, 1), pt, inflight))


def gen_latency_ms(env: Env, ev: Evaluation) -> float | None:
    """The generation latency the SLO is about: ITL p95 (one loaded step) or TPOT p95 (mean step / a of the 5 %
    least accepted requests)."""
    if ev.loaded is None:
        return None
    return ev.itl_ms if env.metric == "itl" else ev.tpot_ms


# --- Checks -------------------------------------------------------------------

def _status(low: float | None, high: float | None, target: float | None) -> CheckStatus:
    """ok if even the pessimistic value meets the target, ko if even the optimistic one misses it."""
    if low is None or high is None or target is None:
        return "unknown"
    if high <= target:
        return "ok"
    return "ko" if low > target else "at_risk"


def _measured_status(value: float, target: float | None) -> CheckStatus:
    return "unknown" if target is None else ("ok" if value <= target else "ko")


def _range_text(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    if math.isinf(high):
        return f"{_ms(low)} – ∞"
    return f"{_ms(low)} – {_ms(high)}"


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    if math.isinf(value):
        return "∞"
    return f"{fmt_int(value)} ms" if value >= 100 else f"{value:.1f} ms"


def checks_for(env: Env, ev: Evaluation, observed=None) -> list[SloCheck]:
    ctx = env.ctx
    slo = ctx.request.slo
    inflight = ev.per_instance if ev.per_instance is not None else None
    lo = evaluate(with_factor(env, 0), ev.point, inflight) if env.model else ev
    hi = evaluate(with_factor(env, 1), ev.point, inflight) if env.model else ev
    rho_max = ctx.heuristic("prefill_load_max").value
    kv_max = ctx.heuristic("kv_usage_max").value
    out: list[SloCheck] = []

    if ev.kv_usage is not None:
        kv_status: CheckStatus = "ok" if ev.kv_usage <= kv_max else ("at_risk" if ev.kv_usage <= 1 else "ko")
        out.append(SloCheck(
            key="kv", label="Occupation du KV cache", status=kv_status, target=kv_max * 100, estimate=round(ev.kv_usage * 100, 1),
            unit="%",
            detail=(f"{ev.active} séquences × {fmt_int(env.context_tokens)} tokens (ISL moyen estimé"
                    f" {fmt_int(env.isl_mean)} + OSL moyen / 2, d'après les percentiles)"
                    f" / {fmt_int(env.kv_tokens)} tokens de KV = {ev.kv_usage:.0%}. Préfixes partagés non déduits (prudent)."),
        ))
    rho_status: CheckStatus = "ko" if ev.rho >= 1 else ("at_risk" if ev.rho > rho_max else "ok")
    out.append(SloCheck(
        key="prefill_load", label="Charge de prefill du budget", status=rho_status, target=rho_max * 100,
        estimate=None if math.isinf(ev.rho) else round(ev.rho * 100, 1), unit="%",
        detail=(
            "Le decode seul remplit le budget : aucune place pour le prefill." if ev.capacity <= 0 else
            f"Prefill demandé {ev.active}·{env.a:g}/{fmt_int(env.osl_mean)} × {fmt_int(env.isl_eff_mean)} tokens non cachés"
            f" = {fmt_int(ev.rho * ev.capacity)} tokens par step, pour {fmt_int(ev.capacity)} disponibles"
            f" (B − S·d = {fmt_int(ev.point.budget)} − {ev.active}·{env.d}) : {ev.rho:.0%}."
        ),
    ))
    if ev.per_instance is not None:
        queued = ev.per_instance > ev.point.max_num_seqs
        out.append(SloCheck(
            key="slots", label="Places de séquence (max-num-seqs)", status="at_risk" if queued else "ok",
            in_verdict=not slo.ttft_p95_ms,
            target=ev.point.max_num_seqs, estimate=ev.per_instance, unit="séquences",
            detail=(f"Requêtes en vol par instance {ev.per_instance} "
                    + (f"> max-num-seqs {ev.point.max_num_seqs} : {ev.per_instance - ev.point.max_num_seqs} requêtes attendent"
                       " une place, ce qui s'ajoute au TTFT." if queued else f"≤ max-num-seqs {ev.point.max_num_seqs}.")),
        ))

    itl_measured = observed.itl_p95_ms if observed else None
    gen_label = "ITL p95" if env.metric == "itl" else "TPOT p95"
    central, low, high = gen_latency_ms(env, ev), gen_latency_ms(env, lo), gen_latency_ms(env, hi)
    if slo.itl_p95_ms or central is not None:
        status = (_measured_status(itl_measured, slo.itl_p95_ms) if itl_measured
                  else _status(low, high, slo.itl_p95_ms))
        detail = "Pas de modèle de latence." if ev.loaded is None else (
            f"Plage {_range_text(low, high)} (incertitude ×{env.spread[0]:g} à ×{env.spread[1]:g})."
            f" Step chargé ≈ {_ms(ev.loaded.ms)} ({ev.active} séquences × {env.k + 1} tokens vérifiés + prefill),"
            f" decode pur ≈ {_ms(ev.decode.ms)}, borné par la "
            + ("mémoire (lecture des poids et du KV)." if ev.loaded.bound == "memory" else "puissance de calcul.")
            + (f" Le step chargé se décompose en {_ops_text(ev.loaded)}." if ev.loaded.ops else "")
            + (f" ITL = un step ; TPOT p95 = step moyen {_ms(ev.average.ms)} / a des 5 % de requêtes les moins acceptées"
               f" ({env.a_low:.2f}, pour a moyen = {env.a:g}) ≈ {_ms(ev.tpot_ms)}." if env.k else
               f" TPOT p95 = step moyen ≈ {_ms(ev.tpot_ms)}.")
        )
        out.append(SloCheck(key="itl", label=gen_label, status=status, target=slo.itl_p95_ms,
                            estimate=None if central is None else round(central, 1),
                            estimate_low=None if low is None else round(low, 1),
                            estimate_high=None if high is None else round(high, 1),
                            measured=itl_measured, unit="ms", detail=detail))
    ttft_measured = observed.ttft_p95_ms if observed else None
    if slo.ttft_p95_ms or ev.ttft_ms is not None:
        status = (_measured_status(ttft_measured, slo.ttft_p95_ms) if ttft_measured
                  else _status(lo.ttft_ms, hi.ttft_ms, slo.ttft_p95_ms))
        detail = "Pas de modèle de latence." if ev.chunk_step is None else (
            f"Plage {_range_text(lo.ttft_ms, hi.ttft_ms)} (incertitude du prefill ×{env.spread_prefill[0]:g} à"
            f" ×{env.spread_prefill[1]:g}). 95e centile de la somme frontal + attente d'une place + attente du prefill"
            f" + prefill du prompt, sur la distribution des tokens non cachés ({env.cold_fraction:.1%} de requêtes"
            f" froides prefillées en entier, hit effectif {env.hit_effective:.1%}). Composantes au p95 : frontal"
            f" {_ms(ev.frontend_s * 1e3)}, attente d'une place {_ms(ev.slot_wait_s * 1e3)}, attente du prefill"
            f" {_ms(ev.prefill_wait_p95_s * 1e3)} (moyenne {_ms(ev.prefill_wait_s * 1e3)}, file M/G/{ev.servers},"
            f" {ev.p_wait:.0%} des requêtes attendent), prefill de {fmt_int(env.isl_eff_p95)} tokens non cachés en"
            f" morceaux de {fmt_int(ev.chunk)} : {_ms(ev.own_prefill_s * 1e3)}."
        )
        out.append(SloCheck(key="ttft", label="TTFT p95", status=status, target=slo.ttft_p95_ms,
                            estimate=None if ev.ttft_ms is None else round(ev.ttft_ms, 1),
                            estimate_low=None if lo.ttft_ms is None else round(lo.ttft_ms, 1),
                            estimate_high=None if hi.ttft_ms is None else round(hi.ttft_ms, 1),
                            measured=ttft_measured, unit="ms", detail=detail))
        if ev.chunk_step is not None:
            out.append(SloCheck(
                key="ttft_cold", label="TTFT d'une reprise à froid (ISL P95 sans cache)",
                status=_status(lo.ttft_cold_ms, hi.ttft_cold_ms, slo.ttft_p95_ms), target=slo.ttft_p95_ms,
                estimate=round(ev.ttft_cold_ms, 1), estimate_low=round(lo.ttft_cold_ms, 1),
                estimate_high=round(hi.ttft_cold_ms, 1), unit="ms", in_verdict=False,
                detail=(f"Cas pire, hors verdict (les requêtes froides sont déjà dans le TTFT p95, à leur fréquence) :"
                        f" prompt ISL P95 entièrement hors cache, {fmt_int(env.isl_cold)} tokens à recalculer en"
                        f" {math.ceil(env.isl_cold / ev.chunk)} morceaux, attention sur tout le prompt comprise."),
            ))
    front = ev.frontend
    if env.model is not None and ev.loaded is not None:
        if front is None:
            out.append(SloCheck(key="frontend", label="Frontal API (CPU)", status="unknown", unit="%",
                                detail="Frontend Rust : coût CPU non modélisé, comparer les deux frontends sur un palier."))
        else:
            umax = env.frontend_costs.utilization_max
            util = front.utilization
            shares = ", ".join(f"{name} {share:.0%}" for name, share in front.gil_share.items())
            out.append(SloCheck(
                key="frontend", label="Frontal API (CPU, frontend Python)",
                status="ko" if front.saturated else ("at_risk" if util > umax else "ok"),
                target=umax * 100, estimate=None if math.isinf(util) else round(util * 100, 1), unit="%",
                detail=(f"{front.api_servers} processus API × {front.workers} thread(s) de renderer pour"
                        f" {fmt_int(front.requests_per_s)} requêtes/s et {fmt_int(front.chunks_per_s)} sorties streamées/s :"
                        f" GIL de chaque processus à {front.gil_utilization:.0%} ({shares}), pool du renderer à"
                        f" {front.pool_utilization:.0%}, ~{front.cores:.1f} cœur(s). Attentes : {_ms(front.loop_wait_p95_s * 1e3)}"
                        f" au p95 sur le GIL (ajoutées à l'ITL), {_ms(front.pool_wait_s * 1e3)} en moyenne pour le rendu du"
                        " prompt (ajoutées au TTFT). Coûts CPU supposés (🔶)."),
            ))
    return out


def worst(checks: list[SloCheck]) -> CheckStatus:
    statuses = [c.status for c in checks if c.in_verdict and c.status != "unknown"]
    return max(statuses, key=RANK.get) if statuses else "unknown"


def _structural_ok(env: Env, pt: Point, per_instance: int | None, rho: float, kv_usage: float | None) -> bool:
    """KV, prefill load and sequence slots all « ok »."""
    if kv_usage is not None and kv_usage > env.ctx.heuristic("kv_usage_max").value:
        return False
    if rho >= 1 or rho > env.ctx.heuristic("prefill_load_max").value:
        return False
    return not (per_instance is not None and per_instance > pt.max_num_seqs and not env.ctx.request.slo.ttft_p95_ms)


def _latency_ok(env: Env, hi: Evaluation) -> bool:
    """Pessimistic generation latency and TTFT within their targets (or not estimated)."""
    slo = env.ctx.request.slo
    if hi.loaded is None:
        return True
    if slo.itl_p95_ms and gen_latency_ms(env, hi) > slo.itl_p95_ms:
        return False
    return not (slo.ttft_p95_ms and hi.ttft_ms is not None and hi.ttft_ms > slo.ttft_p95_ms)


def _frontend_ok(env: Env, ev: Evaluation) -> bool:
    """Frontend (central estimate) under its utilization threshold, or not modelled."""
    front = ev.frontend
    return front is None or env.frontend_costs is None or front.utilization <= env.frontend_costs.utilization_max


def passes(env: Env, ev: Evaluation) -> bool:
    """worst(checks_for(env, ev)) == "ok", evaluating only the pessimistic end (what « ok » requires)."""
    if not _structural_ok(env, ev.point, ev.per_instance, ev.rho, ev.kv_usage):
        return False
    if env.model is None or ev.loaded is None:
        return True
    return _frontend_ok(env, ev) and _latency_ok(env, evaluate(with_factor(env, 1), ev.point, ev.per_instance))


def frontend_changes(env: Env, ev: Evaluation) -> dict[str, Any]:
    """Frontend flags of an operating point found with a re-sized frontend, when they differ from the config."""
    front, changes = ev.frontend, {}
    if front is None:
        return changes
    if front.api_servers != env.api_servers:
        changes["--api-server-count"] = front.api_servers
    if front.workers != env.renderer_workers:
        changes["--renderer-num-workers"] = front.workers
    return changes


def _ops_text(step: p.StepTime) -> str:
    labels = {"dense": "poids denses", "experts": "experts", "attention": "attention", "allreduce": "all-reduce TP",
              "fixed": "surcoûts fixes", "drafter": "drafter"}
    return ", ".join(f"{labels.get(key, key)} {_ms(value * 1e3)}" for key, value in step.ops.items() if value > 0)


# --- Search -------------------------------------------------------------------

def kv_seq_cap(env: Env) -> int | None:
    if not env.kv_tokens:
        return None
    return max(1, math.floor(env.kv_tokens * env.ctx.heuristic("kv_usage_max").value / env.context_tokens))


def candidate_points(env: Env, base: Point, instances: int) -> list[Point]:
    wl = env.wl
    per_instance = math.ceil(wl.target_concurrency / instances) if wl.target_concurrency else None
    cap = kv_seq_cap(env)
    seqs = {base.max_num_seqs, 16, 32, 64, 128, 192, 256, 384, 512}
    seqs |= {max(1, base.max_num_seqs * n // 8) for n in range(1, 8)}
    if per_instance:
        seqs.add(per_instance)
    if cap:
        seqs = {s for s in seqs if s <= cap} or {cap}
    budgets = sorted(set(BUDGET_GRID) | {base.budget})
    points = []
    for s in sorted(seqs):
        for b in budgets:
            if env.dspark_cap and s > b // (env.k + 1):
                continue
            for t in sorted({b // 4, b // 2} | ({base.threshold} if base.threshold <= b else set())):
                points.append(Point(max_num_seqs=s, budget=b, threshold=t, instances=instances))
    return points


def best_point(env: Env, points: list[Point], base: Point) -> tuple[Point, Evaluation] | None:
    """Among the points that pass every check: fewest parameter changes from the base, then highest throughput."""
    best = None
    pessimistic = with_factor(env, 1)
    for pt in points:
        per_instance, _, capacity, _, rho, kv_usage = _load(env, pt, None)
        if not _structural_ok(env, pt, per_instance, rho, kv_usage):
            continue
        if env.model is not None and capacity > 0 and not _latency_ok(env, evaluate(pessimistic, pt)):
            continue
        ev = evaluate(env, pt)
        if not _frontend_ok(env, ev):
            continue
        changed = sum((pt.max_num_seqs != base.max_num_seqs, pt.budget != base.budget, pt.threshold != base.threshold))
        tput = ev.throughput(env.a) or ev.active
        key = (-changed, round(tput, -2), -pt.budget)
        if best is None or key > best[0]:
            best = (key, pt, ev)
    return (best[1], best[2]) if best else None


def changes_between(env: Env, base: Point, new: Point) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    if new.max_num_seqs != base.max_num_seqs:
        changes["--max-num-seqs"] = new.max_num_seqs
        size = f.clamp_capture_size(f.cudagraph_capture_size(env.k, new.max_num_seqs), new.budget, env.k + 1)
        changes["--compilation-config"] = {"max_cudagraph_capture_size": size}
    if new.budget != base.budget:
        changes["--max-num-batched-tokens"] = new.budget
        if "--compilation-config" not in changes:
            size = f.clamp_capture_size(f.cudagraph_capture_size(env.k, new.max_num_seqs), new.budget, env.k + 1)
            changes["--compilation-config"] = {"max_cudagraph_capture_size": size}
    if new.threshold != base.threshold:
        changes["--long-prefill-token-threshold"] = new.threshold
    return changes


def _describe(changes: dict[str, Any]) -> str:
    return ", ".join(f"{flag} {value['max_cudagraph_capture_size'] if isinstance(value, dict) else value}"
                     .replace("--compilation-config", "capture CUDA graphs") for flag, value in changes.items())


# --- Bottlenecks ----------------------------------------------------------------

def diagnose(env: Env, base: Point, ev: Evaluation, checks: list[SloCheck]) -> tuple[list[Bottleneck], list[Recommendation]]:
    ctx = env.ctx
    by = {c.key: c for c in checks}
    slo = ctx.request.slo
    bottlenecks: list[Bottleneck] = []
    recos: list[Recommendation] = []
    bad = lambda key: key in by and by[key].status in ("at_risk", "ko")  # noqa: E731

    if ev.capacity <= 0:
        bottlenecks.append(Bottleneck(
            key="budget_decode", label="Budget saturé par le decode", status="ko",
            explanation=(f"{ev.active} séquences × d = {env.d} tokens = {fmt_int(ev.active * env.d)} ≥ budget de"
                         f" {fmt_int(base.budget)} : plus aucun prefill ne passe, les nouvelles requêtes n'avancent plus."),
        ))
    if bad("kv"):
        bottlenecks.append(Bottleneck(
            key="kv_capacity", label="KV cache", status=by["kv"].status,
            explanation=(f"Le KV ne loge pas confortablement les {ev.active} séquences actives ({by['kv'].estimate:g} %"
                         " d'occupation moyenne) : vLLM préempte des séquences (recalcul, pics de TTFT et d'ITL)."),
        ))
        current_util = ctx.request.observed_kv.gpu_memory_utilization if ctx.request.observed_kv else None
        if (current_util or 0) < 0.9:
            recos.append(Recommendation(
                kind="param", title="Relever --gpu-memory-utilization (si aucun OOM à la capture des CUDA graphs)",
                rationale=(f"Chaque point de gpu-memory-utilization ajoute ~{ctx.gpu.memory.hbm_gb * 0.01:.1f} Go de KV par GPU."
                           " Valeur de 0,8 retenue pour protéger la capture : passer à 0,85–0,9 puis relire « GPU KV cache size »."),
                changes={"--gpu-memory-utilization": 0.9}, addresses=["kv_capacity"],
            ))
        if ctx.model.kv_cache.replicated_across_tp:
            recos.append(Recommendation(
                kind="architecture", title="DP-attention (data parallel dans l'instance)",
                rationale=("Le KV de ce modèle est répliqué sur chaque rang TP : en DP-attention, chaque rang garde son propre"
                           f" KV, soit jusqu'à ×{ctx.request.gpus_per_instance} de contextes logeables (§5). À valider sur la recette du modèle."),
                addresses=["kv_capacity"],
            ))
        if ctx.request.workload.paused_sessions:
            recos.append(Recommendation(
                kind="param", title="Offload du KV vers la RAM hôte",
                rationale="Des sessions en pause occupent le KV : --kv-offloading-size les garde en RAM hôte sans bloquer les actives.",
                addresses=["kv_capacity"],
            ))

    if bad("prefill_load") or (bad("ttft") and ev.prefill_wait_p95_s >= max(ev.slot_wait_s, ev.own_prefill_s, ev.frontend_s)):
        bottlenecks.append(Bottleneck(
            key="prefill", label="Prefill (débit du budget de tokens)", status=by["prefill_load"].status if bad("prefill_load") else by["ttft"].status,
            explanation=(
                "Le prefill non caché arrive plus vite que le budget ne peut l'absorber : les requêtes attendent avant leur premier"
                " token et la file grandit." if ev.rho >= 1 else
                f"Le budget de prefill est chargé à {ev.rho:.0%} : l'attente avant prefill ({_ms(ev.prefill_wait_p95_s * 1e3)}"
                " au p95) domine le TTFT. Les prompts froids, longs, occupent les voies de prefill (file M/G/c à forte"
                " variance)."
            ),
        ))
        recos.append(Recommendation(
            kind="architecture", title="Augmenter le prefix cache hit rate",
            rationale=(f"Chaque point de hit rate retire {fmt_int(env.isl_mean * 0.01)} tokens de prefill par requête."
                       " Prompts stables en tête (system prompt, outils), routage par affinité de session (EPP llm-d)."),
            addresses=["prefill"],
        ))
        recos.append(Recommendation(
            kind="architecture", title="Désagrégation prefill / decode si aucun budget ne concilie TTFT et ITL",
            rationale="Les pods de prefill absorbent les gros prompts sans allonger les steps des séquences en decode (§5).",
            addresses=["prefill"],
        ))

    if bad("ttft") and ev.chunk_step is not None:
        parts = {"slots": ev.slot_wait_s, "prefill": ev.prefill_wait_p95_s, "chunks": ev.own_prefill_s,
                 "frontend": ev.frontend_s}
        main = max(parts, key=parts.get)
        if main == "slots":
            bottlenecks.append(Bottleneck(
                key="slots", label="Places de séquence (max-num-seqs)", status=by["ttft"].status,
                explanation=(f"{ev.per_instance - base.max_num_seqs} requêtes par instance attendent une place :"
                             f" {_ms(ev.slot_wait_s * 1e3)} d'attente, l'essentiel du TTFT."),
            ))
        elif main == "frontend":
            bottlenecks.append(Bottleneck(
                key="frontend", label="Frontal (HTTP, routage, tokenisation)", status=by["ttft"].status,
                explanation=(f"Avant le GPU, ≈ {_ms(ev.frontend_s * 1e3)} par prompt au p95 (tokenisation, chat template,"
                             " passage par l'EPP) : comparer le TTFT client au TTFT serveur (vllm:time_to_first_token_seconds)."),
            ))
        elif main == "chunks":
            bottlenecks.append(Bottleneck(
                key="prefill_chunks", label="Prefill d'une longue requête (seuil de long prefill)", status=by["ttft"].status,
                explanation=(f"Le p95 des tokens à prefiller ({fmt_int(env.isl_eff_p95)} tokens non cachés"
                             + (f", tiré par les {env.cold_fraction:.1%} de requêtes froides" if env.cold_fraction else "")
                             + f") est découpé en morceaux de {fmt_int(ev.chunk)} :"
                             f" {max(1, math.ceil(env.isl_eff_p95 / ev.chunk))} steps de {_ms(ev.chunk_step.ms)}."),
            ))
            if env.cold_fraction:
                recos.append(Recommendation(
                    kind="architecture", title="Réduire les requêtes froides",
                    rationale=("Chaque requête dont le préfixe manque au cache recalcule tout son prompt : routage par affinité"
                               " de session (EPP llm-d, prefix-cache-scorer), préfixe commun stable en tête, offload du KV"
                               " des sessions en pause plutôt que leur éviction."),
                    addresses=["prefill_chunks"],
                ))
            if version_key(ctx.vllm.version) >= (0, 33, 0) and ctx.model.architecture.model_type.startswith("deepseek"):
                recos.append(Recommendation(
                    kind="architecture", title="Prefill context parallel (PCP) pour les très longs prompts",
                    rationale="Disponible en 0.33 avec le prefix caching sur DeepSeek V4.1 (#59857) : réduit le TTFT des gros prefills.",
                    addresses=["prefill_chunks"],
                ))

    if bad("itl") and ev.loaded is not None:
        single = p.step_time(env.model, 1, env.k + 1, env.context_tokens, 0)
        target = env.step_target_ms
        if target and single.ms > ctx.heuristic("slo_margin").value["ok_below"] * target:
            bottlenecks.append(Bottleneck(
                key="itl_floor", label="Plancher d'ITL du déploiement", status=by["itl"].status,
                explanation=(f"Même avec une seule séquence, un step dure ≈ {_ms(single.ms)} (dont {_ms(single.overhead_s * 1e3)}"
                             " de surcoût fixe) : ni max-num-seqs ni des instances supplémentaires ne suffiront."),
            ))
            recos.append(Recommendation(
                kind="architecture", title="Augmenter le tensor parallel ou alléger les poids",
                rationale=(f"Passer à TP{ctx.request.gpus_per_instance * 2} divise par 2 les octets lus par GPU ;"
                           " un checkpoint plus compact (NVFP4) réduit aussi la lecture des poids. Si la cible porte sur le temps"
                           " par token (TPOT) et non sur l'intervalle entre chunks, elle se compare à step / a."),
                addresses=["itl_floor"],
            ))
        elif ev.decode.ms > ctx.heuristic("slo_margin").value["ok_below"] * (target or math.inf):
            limit = ctx.heuristic("slo_margin").value["ok_below"] * target
            s_max = max((s for s in range(1, ev.active + 1)
                         if p.step_time(env.model, s, env.k + 1, env.context_tokens, 0).ms <= limit), default=1)
            explanation = (f"Le decode pur de {ev.active} séquences × {env.k + 1} tokens dure déjà ≈ {_ms(ev.decode.ms)},"
                           f" borné par la {'mémoire' if ev.decode.bound == 'memory' else 'puissance de calcul'}."
                           f" Il faut descendre à ~{s_max} séquences par instance pour tenir la cible.")
            if env.model.routed_params and s_max * 8 < ev.active:
                explanation += (" La lecture des experts MoE varie peu avec le nombre de séquences : réduire la concurrence par"
                                " instance coûte beaucoup de GPU pour peu de gain ; un TP plus large ou des poids plus compacts"
                                " sont plus efficaces.")
            bottlenecks.append(Bottleneck(key="decode", label="Decode (trop de séquences par step)", status=by["itl"].status,
                                          explanation=explanation))
            if s_max * 8 < ev.active:
                recos.append(Recommendation(
                    kind="architecture", title="Augmenter le tensor parallel ou alléger les poids",
                    rationale=(f"TP{ctx.request.gpus_per_instance * 2} divise par 2 les octets lus par GPU à chaque step ;"
                               " un checkpoint plus compact (NVFP4) réduit aussi la lecture des poids."),
                    addresses=["decode"],
                ))
        else:
            bottlenecks.append(Bottleneck(
                key="prefill_interference", label="Interférence du prefill sur le decode", status=by["itl"].status,
                explanation=(f"Le decode pur tient ({_ms(ev.decode.ms)}), mais les steps qui embarquent du prefill durent"
                             f" ≈ {_ms(ev.loaded.ms)} : un budget plus bas ou un seuil de long prefill plus bas lissent l'ITL."),
            ))
    if bad("frontend") and ev.frontend is not None:
        front, costs = ev.frontend, env.frontend_costs
        main = max(front.gil_share, key=front.gil_share.get)
        gil_bound = front.gil_utilization >= front.pool_utilization
        bottlenecks.append(Bottleneck(
            key="frontend_cpu", label="Frontal API (CPU du frontend Python)", status=by["frontend"].status,
            explanation=(
                (f"Le GIL de chaque processus API est occupé à {front.gil_utilization:.0%}, surtout par {main}"
                 f" ({front.gil_share[main]:.0%}) : les sorties partent en retard (ITL) et les requêtes attendent leur rendu"
                 " (TTFT). Seuls d'autres processus API (--api-server-count) le soulagent ; le pool du renderer n'y peut"
                 " rien." if gil_bound else
                 f"Le pool du renderer est occupé à {front.pool_utilization:.0%} : les requêtes attendent leur chat template"
                 " et leur tokenisation, le TTFT monte. Plus de threads (--renderer-num-workers) pour la tokenisation, plus"
                 " de processus API si le chat template domine.")
            ),
        ))
        servers, workers = fe.size_frontend(costs, front.requests_per_s, front.chunks_per_s, env.isl_mean,
                                            env.max_api_servers)
        changes = {}
        if servers > front.api_servers:
            changes["--api-server-count"] = servers
        if workers > front.workers:
            changes["--renderer-num-workers"] = workers
        if changes:
            recos.append(Recommendation(
                kind="param", title="Dimensionner le frontal : " + ", ".join(f"{k} {v}" for k, v in changes.items()),
                rationale=(f"Ramène chaque GIL et chaque pool sous {costs.utilization_max:.0%}. Prévoir ~1 cœur par processus"
                           " API en plus et vérifier la limite CPU du pod : un CPU bridé ralentit aussi les workers GPU."),
                changes=changes, addresses=["frontend_cpu"],
            ))
        else:
            recos.append(Recommendation(
                kind="capacity", title="Plus de CPU par pod, ou plus d'instances",
                rationale=(f"Le nombre de processus API est plafonné ({env.max_api_servers}) : au-delà, répartir la charge sur"
                           " plus d'instances (chacune a son frontal) ou relever la limite CPU du pod."),
                addresses=["frontend_cpu"],
            ))
        recos.append(Recommendation(
            kind="architecture", title="Comparer le frontend Rust (VLLM_USE_RUST_FRONTEND=1)",
            rationale=("Frontend multithread, sans GIL : --api-server-count n'y a plus de sens (vLLM le ramène à 1). Son coût"
                       " CPU n'est pas modélisé ici : à mesurer en A/B sur le même palier, en vérifiant d'abord que tous"
                       " les flags de la config sont acceptés (il refuse les flags inconnus)."),
            addresses=["frontend_cpu"],
        ))
    bottlenecks.sort(key=lambda b: -RANK[b.status])
    return bottlenecks, recos


# --- Entry point ---------------------------------------------------------------

def build_env(ctx: RuleContext, state: dict[str, Any], kv_tokens: int | None,
              model: p.PerfModel | None) -> tuple[Env, list[str]]:
    """Everything that does not change between operating points, including the distribution of the tokens to prefill."""
    req, wl = ctx.request, ctx.request.workload
    h = lambda key: ctx.heuristic(key).value  # noqa: E731
    hit = state["prefix_cache_hit_rate"]
    cap = state.get("max_model_len") or ctx.model.architecture.max_position_embeddings
    quantile = f.lognormal_quantile_fn(wl.isl.p50, wl.isl.p95, wl.isl.p99, cap=cap)
    shared = (req.routing.shared_prefix_tokens or 0) if req.routing else 0
    cold_h = h("cold_request_fraction")
    notes = []
    probe, _ = f.uncached_distribution(quantile, 0.0, 0.0, 0.0)
    mean_isl = sum(prob * isl for prob, isl, _ in probe)
    mean_cold = sum(prob * max(isl - shared, 1.0) for prob, isl, _ in probe)
    if hit <= 0:
        cold = 1.0
    elif wl.cold_request_fraction is not None:
        cold = wl.cold_request_fraction
    else:
        # Défaut plafonné : les requêtes froides ne portent pas plus d'une part donnée du prefill non caché, sans quoi le
        # hit rate déclaré serait intenable (les premiers tours de session sont souvent plus courts que la moyenne).
        ceiling = cold_h["max_share_of_uncached_tokens"] * (1 - hit) * mean_isl / mean_cold if mean_cold else 0.0
        cold = min(cold_h["value"], ceiling)
        notes.append(
            f"Requêtes froides (préfixe absent du cache : nouvelle session au-delà du préfixe commun, éviction, mauvais"
            f" routage) : {cold:.1%} des requêtes, hypothèse 🔶 ({cold_h['value']:.0%}"
            + (f", plafonnée pour que leur prefill ne dépasse pas {cold_h['max_share_of_uncached_tokens']:.0%} du prefill"
               f" non caché d'un hit rate de {hit:.1%}" if cold < cold_h["value"] else "")
            + "). Mesure : part des requêtes dont vllm:prefix_cache_hits est nul ; saisir « Part de requêtes froides »."
        )
    points, hit_eff = f.uncached_distribution(quantile, hit, cold if hit > 0 else 0.0, h("uncached_tokens_sigma"),
                                              shared)
    dist = tuple(f.collapse(points))
    if hit > 0 and hit_eff < hit - 0.002:
        notes.append(f"Les requêtes froides ({cold:.1%}) prefillent à elles seules plus que ne le permet le hit rate de"
                     f" {hit:.1%} : hit effectif retenu {hit_eff:.1%} (prudent). Vérifier l'une des deux valeurs.")
    k, a = state["num_speculative_tokens"], state["acceptance_length"]
    a_low = f.acceptance_quantile(a, k, 0.05, h("acceptance_dispersion")) if k else 1.0
    caps = frontend_caps(ctx, state)
    costs = None if caps.rust else fe.FrontendCosts.from_heuristic(h("frontend_python"))
    spread = tuple(ctx.heuristic("perf_uncertainty").value["uncalibrated"])
    env = Env(
        ctx=ctx, k=k, d=state["decode_tokens_per_seq"], a=a, hit=hit, kv_tokens=kv_tokens, model=model,
        dspark_cap=bool(ctx.behavior("spec_decode.dspark_active_seq_cap").value) and state.get("spec_method") in ("dspark", "dflash"),
        dist=dist, hit_effective=hit_eff, cold_fraction=cold if 0 < hit else 0.0, a_low=a_low,
        frontend_costs=costs, api_servers=caps.api_servers, renderer_workers=caps.renderer_workers,
        max_api_servers=caps.max_servers, spread=spread, spread_prefill=spread,
    )
    notes.append(
        f"Tokens à prefiller par requête : moyenne {fmt_int(env.isl_eff_mean)}, p95 {fmt_int(env.isl_eff_p95)}, p99"
        f" {fmt_int(f.quantile_of_points(list(dist), 0.99))} (ISL lognormale ajustée sur les percentiles"
        + (f", {env.cold_fraction:.1%} de requêtes froides prefillées en entier, part non cachée des requêtes chaudes"
           f" dispersée (σ = {h('uncached_tokens_sigma'):g}, sorties d'outils)" if 0 < hit else ", aucun cache")
        + f"). L'ancien calcul ISL P95 × (1 − hit) donnait {fmt_int((wl.isl.p95 or wl.isl.p50) * (1 - hit))}."
    )
    if k:
        notes.append(f"Longueur d'acceptation : moyenne a = {a:g}, {a_low:.2f} pour les 5 % de requêtes les moins acceptées"
                     f" (dispersion entre requêtes, concentration {h('acceptance_dispersion'):g} 🔶) : c'est elle qui fixe le"
                     " TPOT p95.")
    if costs is None:
        notes.append("Frontend Rust (VLLM_USE_RUST_FRONTEND=1) : son coût CPU n'est pas modélisé, le TTFT et l'ITL ne"
                     " comptent ni le réseau, ni le rendu du prompt, ni l'attente du frontal.")
    else:
        notes.append(
            f"Frontal Python : {fmt_int(costs.network_s * 1e3)} ms de réseau et d'EPP, puis par requête"
            f" {costs.request_loop_s * 1e3:g} ms de boucle asyncio et {costs.render_s(1000) * 1e3 - costs.request_loop_s * 1e3:.2f} ms"
            " par millier de tokens de prompt (JSON des messages, chat template, tokenisation) ; chaque sortie streamée"
            f" coûte {costs.chunk_s * 1e6:g} µs de boucle (🔶). {caps.api_servers} processus API ×"
            f" {caps.renderer_workers} thread(s) de renderer ; les attentes sur le GIL et le pool s'ajoutent au TTFT et à"
            " l'ITL. Plus un demi-step d'attente de l'ordonnancement."
        )
    notes += caps.notes
    return env, notes


@dataclass(frozen=True)
class FrontendCaps:
    rust: bool
    api_servers: int        # valeur en vigueur : décision de l'app, argument supplémentaire ou défaut vLLM
    renderer_workers: int
    max_servers: int        # plafond des processus API (frontend Rust, LoRA à chaud, Elastic EP, CPU du pod)
    reasons: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def frontend_caps(ctx: RuleContext, state: dict[str, Any]) -> FrontendCaps:
    """Frontend kind, values in force and ceiling of the API server processes for this deployment."""
    req = ctx.request
    extra = {(k if k.startswith("--") else "--" + k.lstrip("-")): v for k, v in req.extra_args.items()}
    env_vars = {**state.get("env", {}), **req.extra_env}
    rust = _truthy(env_vars.get("VLLM_USE_RUST_FRONTEND", "0"))
    beh = ctx.behavior("frontend.api_servers").value
    reasons: list[str] = []
    max_servers = 64
    if rust:
        max_servers = 1
        reasons.append("frontend Rust : vLLM ignore --api-server-count et le ramène à 1")
    for name in beh.get("incompatible_env", []):
        if _truthy(env_vars.get(name, "0")):
            max_servers = 1
            reasons.append(f"{name}=1 est incompatible avec plusieurs processus API")
    if beh.get("elastic_ep_caps_to_one") and _truthy(extra.get("--enable-elastic-ep", False)):
        max_servers = 1
        reasons.append("Elastic EP plafonne vLLM à un processus API")
    h = ctx.heuristic("frontend_python").value
    tp = req.gpus_per_instance
    if req.cpu_cores_per_instance:
        spare = req.cpu_cores_per_instance - h["cores_engine_core"] - tp * h["cores_per_gpu_worker"]
        cpu_cap = max(1, math.floor(spare))
        if cpu_cap < max_servers:
            max_servers = cpu_cap
            reasons.append(f"{req.cpu_cores_per_instance:g} cœurs par instance, dont {h['cores_engine_core']:g} pour l'engine"
                           f" core et {tp * h['cores_per_gpu_worker']:g} pour les workers GPU")
    dp = state.get("data_parallel_size") or 1
    servers = state.get("api_server_count") or (1 if rust else dp)
    workers = state.get("renderer_num_workers") or ctx.default("renderer_num_workers") or 1
    notes: list[str] = []
    if "--api-server-count" in extra:
        servers = 1 if rust else int(extra["--api-server-count"])
        notes.append(f"--api-server-count {extra['--api-server-count']} forcé en argument supplémentaire : retenu par l'analyse.")
    if "--renderer-num-workers" in extra:
        workers = int(extra["--renderer-num-workers"])
        notes.append(f"--renderer-num-workers {workers} forcé en argument supplémentaire : retenu par l'analyse.")
    return FrontendCaps(rust=rust, api_servers=max(1, servers), renderer_workers=max(1, workers),
                        max_servers=max_servers, reasons=tuple(reasons), notes=tuple(notes))


CALIBRATION_BOUNDS = (0.2, 10.0)


def calibrate(env: Env, base: Point, observed, uncertainty: dict) -> tuple[Env, float | None, float | None, list[str]]:
    """Recalibrate the decode part on the measured ITL/TPOT p95 and the prefill part on the measured TTFT p95.

    The step durations are linear in the two factors (the token counts do not depend on them), so the generation
    latency is solved exactly. Without a TTFT measurement the prefill part takes the decode factor (same hardware)
    but keeps the uncalibrated uncertainty. With both, the TTFT fixes the prefill factor and the decode factor is
    solved again so that the generation latency still matches. Factors are kept within CALIBRATION_BOUNDS: beyond,
    the gap does not come from the speed of the GPU but from the shape of the load. Returns (env, decode factor,
    prefill factor, notes)."""
    model = env.model
    if model is None or not observed or not (observed.itl_p95_ms or observed.ttft_p95_ms):
        return env, None, None, []
    lo_bound, hi_bound = CALIBRATION_BOUNDS
    notes: list[str] = []
    decode = prefill = 1.0
    calibration = calibration_prefill = None
    spread, spread_prefill = env.spread, env.spread_prefill

    def at(decode_factor: float, prefill_factor: float) -> Evaluation:
        return evaluate(replace(env, model=model.calibrated(decode_factor, prefill_factor)), base)

    def solve_decode(prefill_factor: float | None) -> float | None:
        """Decode factor matching the measured generation latency (prefill_factor None: tied to it)."""
        ev = at(1.0, 1.0 if prefill_factor is None else prefill_factor)
        if ev.loaded is None:
            return None
        step = ev.loaded if env.metric == "itl" else ev.average
        lag = ev.frontend.loop_wait_p95_s if (ev.frontend and env.metric == "itl") else 0.0
        target = (observed.itl_p95_ms / 1e3 - lag) * (1.0 if env.metric == "itl" else env.a_low)
        if prefill_factor is None:
            factor = target / step.seconds
        else:
            factor = (target - (step.seconds - step.decode_s)) / step.decode_s if step.decode_s > 0 else lo_bound
        return min(max(factor, lo_bound), hi_bound)

    if observed.itl_p95_ms:
        found = solve_decode(None)
        if found:
            decode = prefill = calibration = found
            spread = tuple(uncertainty["calibrated"])
    if observed.ttft_p95_ms:
        current = at(decode, prefill).ttft_ms
        if current is None or math.isinf(current):
            notes.append("TTFT mesuré non exploité pour recaler le prefill : au point de la config, la file de prefill est"
                         " instable dans le modèle (charge ≥ 100 %), le TTFT estimé est infini quel que soit le facteur.")
        else:
            lo, hi = lo_bound, hi_bound
            for _ in range(40):
                mid = math.sqrt(lo * hi)
                value = at(decode, mid).ttft_ms
                lo, hi = (mid, hi) if value is not None and value < observed.ttft_p95_ms else (lo, mid)
            prefill = calibration_prefill = math.sqrt(lo * hi)
            spread_prefill = tuple(uncertainty["calibrated"])
            if observed.itl_p95_ms:
                found = solve_decode(prefill)
                if found:
                    decode = calibration = found
            reached = at(decode, prefill)
            gap_ttft = reached.ttft_ms / observed.ttft_p95_ms if reached.ttft_ms else 1.0
            gap_gen = (gen_latency_ms(env, reached) or observed.itl_p95_ms) / observed.itl_p95_ms if observed.itl_p95_ms else 1.0
            if not (0.85 <= gap_ttft <= 1.15 and 0.85 <= gap_gen <= 1.15):
                notes.append(
                    f"Mesures non conciliables par la seule vitesse du GPU (au mieux, TTFT {reached.ttft_ms:.0f} ms pour"
                    f" {observed.ttft_p95_ms:g} mesurés" + (f", latence de génération {gen_latency_ms(env, reached):.1f} ms"
                                                            f" pour {observed.itl_p95_ms:g}" if observed.itl_p95_ms else "")
                    + "). Dans le modèle, le TTFT ≈ morceaux × step chargé + frontal + attente, et le step chargé est"
                    " fixé par la latence de génération : l'écart vient de la forme de la charge (requêtes froides, hit"
                    " rate réel, file d'attente, rafales), pas du prefill. Le prefill n'est pas recalé (le TTFT mesuré"
                    " prime pour le verdict) : mesurer le hit rate et la part de requêtes froides et les saisir."
                )
                if observed.itl_p95_ms and calibration is not None:
                    decode = prefill = calibration = solve_decode(None)
                elif not observed.itl_p95_ms:
                    decode = prefill = 1.0
                calibration_prefill, spread_prefill = None, env.spread_prefill
    env = replace(env, model=model.calibrated(decode, prefill), spread=spread, spread_prefill=spread_prefill)
    label = "ITL" if env.metric == "itl" else "TPOT"
    if calibration is not None:
        notes.append(
            f"Partie decode recalée sur la {label} p95 mesurée ({observed.itl_p95_ms:g} ms) : × {calibration:.2f}."
            + (" Écart important : vérifier que la mesure porte bien sur cette config." if not 0.5 <= calibration <= 2 else "")
            + (" En TPOT, le facteur absorbe aussi l'erreur sur la longueur d'acceptation : saisir a mesuré."
               if env.metric == "tpot" and env.k else "")
        )
    if calibration_prefill is not None:
        notes.append(f"Partie prefill recalée sur le TTFT p95 mesuré ({observed.ttft_p95_ms:g} ms) : × {calibration_prefill:.2f}.")
    elif calibration is not None:
        notes.append("Partie prefill : même facteur que le decode (même matériel), incertitude non réduite faute de TTFT mesuré.")
    return env, calibration, calibration_prefill, notes


def analyze(ctx: RuleContext, state: dict[str, Any]) -> SloAnalysis:
    req = ctx.request
    model, notes = build_perf_model(ctx, state)
    kv_tokens, kv_note = adjusted_kv_tokens(ctx, state)
    if kv_note:
        notes.append(kv_note)
    uncertainty = ctx.heuristic("perf_uncertainty").value
    env, dist_notes = build_env(ctx, state, kv_tokens, model)
    notes += dist_notes
    base = Point(max_num_seqs=state["max_num_seqs"], budget=state["max_num_batched_tokens"],
                 threshold=state["long_prefill_token_threshold"], instances=req.num_instances)
    notes.append(
        "Convention vLLM (vllm/v1/metrics/stats.py) : l'ITL est l'intervalle entre deux sorties d'une requête, soit un step"
        " du scheduler, quel que soit le nombre de tokens de la sortie ; le TPOT divise la durée du decode par le nombre de"
        " tokens. Avec le speculative decoding, l'ITL augmente un peu mais le TPOT est divisé par ~a (moins pour les"
        " requêtes peu acceptées, qui font le TPOT p95)."
    )
    notes.append(
        f"Longueurs moyennes estimées d'après les percentiles : ISL {fmt_int(env.isl_mean)}, OSL {fmt_int(env.osl_mean)}"
        f" (P50 : {fmt_int(req.workload.isl.p50)} et {fmt_int(req.workload.osl.p50)})."
    )
    if not req.workload.target_concurrency:
        notes.append(f"Requêtes en vol non fournies : analyse à pleine charge ({base.max_num_seqs} séquences par instance).")

    env, calibration, calibration_prefill, calib_notes = calibrate(env, base, req.observed_perf, uncertainty)
    notes += calib_notes
    observed = req.observed_perf

    ev = evaluate(env, base)
    checks = checks_for(env, ev, observed)
    verdict = worst(checks)
    bottlenecks, specific = diagnose(env, base, ev, checks)

    recos: list[Recommendation] = []
    slo_given = bool(req.slo.itl_p95_ms or req.slo.ttft_p95_ms)
    if verdict == "ok":
        recos.append(Recommendation(
            kind="none", title="Configuration conforme : aucune action",
            rationale=("Tous les contrôles passent avec marge." if slo_given else
                       "Capacité suffisante. Sans SLO de latence, le verdict ne porte que sur le KV et le prefill.")
            + " Valider en production avec les métriques de l'onglet Vérification.",
        ))
    else:
        # Le frontal se redimensionne point par point : un réglage ou des instances de plus n'en sont pas bloqués.
        search = replace(env, frontend_autosize=True)
        tune = best_point(search, candidate_points(search, base, base.instances), base)
        if tune is not None:
            pt, tev = tune
            changes = changes_between(env, base, pt) | frontend_changes(env, tev)
            if changes:
                recos.append(Recommendation(
                    kind="param", title="Réglage qui tient les SLO sans GPU supplémentaire : " + _describe(changes),
                    rationale=_point_summary(env, tev),
                    changes=changes, addresses=[b.key for b in bottlenecks],
                ))
        if req.workload.target_concurrency:
            tp = req.gpus_per_instance
            scale = _scale_out(search, base)
            wider = _wider_tp(search, base) if model is not None else None
            gpus_now = base.instances * tp
            if scale is not None:
                n, pt, sev = scale
                changes = {"instances": n, "GPU": n * tp} | changes_between(env, base, pt) | frontend_changes(env, sev)
                costly = wider is not None and wider[0] * wider[1] < n * tp
                recos.append(Recommendation(
                    kind="capacity",
                    title=f"Ajouter de la capacité GPU : {_instances(n)} de TP{tp} ({_gpus(n * tp, gpus_now)})",
                    rationale=_point_summary(env, sev)
                    + (" Plus coûteux que l'option TP plus large ci-dessous." if costly else
                       " Option à privilégier : aucun réglage ne suffit à capacité constante." if tune is None else
                       " Alternative au réglage ci-dessus si la marge doit être plus large."),
                    changes=changes, addresses=[b.key for b in bottlenecks],
                ))
            if wider is not None and (scale is None or wider[0] * wider[1] < scale[0] * tp):
                n2, tp2, pt2, ev2 = wider
                recos.append(Recommendation(
                    kind="capacity",
                    title=f"Passer en TP{tp2} : {_instances(n2)} ({_gpus(n2 * tp2, gpus_now)})",
                    rationale=(_point_summary(env, ev2) + f" En TP{tp2}, chaque GPU lit deux fois moins de poids par step :"
                               " c'est le levier quand l'ITL est bornée par la lecture des poids. Hypothèses : même KV par"
                               " instance (prudent), volume d'all-reduce recalculé, même latence fixe par couche."),
                    changes={"instances": n2, "GPU": n2 * tp2, "--tensor-parallel-size": tp2}
                    | changes_between(env, base, pt2) | frontend_changes(env, ev2),
                    addresses=[b.key for b in bottlenecks],
                ))
            if scale is None and wider is None and tune is None:
                notes.append("Aucun nombre d'instances ne tient tous les SLO, même à une séquence par instance :"
                             " voir les pistes d'architecture.")
        else:
            recos.append(Recommendation(
                kind="measure", title="Fournir les requêtes en vol visées pour dimensionner le nombre d'instances",
                rationale="Sans elles, l'analyse suppose des instances pleines et ne peut pas chiffrer la capacité GPU à ajouter.",
            ))
        recos.extend(specific)
    if model is not None and not (observed and observed.itl_p95_ms and observed.ttft_p95_ms):
        recos.append(Recommendation(
            kind="measure", title="Valider par paliers, puis saisir la latence mesurée pour recaler l'analyse",
            rationale=("Le modèle de latence est un ordre de grandeur (×"
                       f"{env.spread[0]:g} à ×{env.spread[1]:g}) : suivre le plan de paliers et saisir la latence de"
                       " génération et le TTFT mesurés au palier de la config générée (chacun recale sa partie du modèle)."),
        ))

    summary = _summary(verdict, bottlenecks, slo_given, model is not None)
    plan = build_test_plan(env, base)
    breakdown = None
    if ev.loaded is not None:
        breakdown = {
            "Step de decode pur (ms)": round(ev.decode.ms, 2),
            "Step chargé, ITL p95 (ms)": round(ev.loaded.ms, 2),
            "Step moyen (ms)": round(ev.average.ms, 2),
            "Borne du step chargé": "mémoire" if ev.loaded.bound == "memory" else "calcul",
            "Opérateurs du step chargé (ms)": _ops_text(ev.loaded),
            "Lecture mémoire / calcul / all-reduce / surcoûts / drafter du step chargé (ms)": (
                f"{ev.loaded.memory_s * 1e3:.2f} / {ev.loaded.compute_s * 1e3:.2f} / {ev.loaded.comm_s * 1e3:.2f}"
                f" / {ev.loaded.overhead_s * 1e3:.2f} / {ev.loaded.drafter_s * 1e3:.2f}"),
            "Tokens à prefiller par requête, moyenne / p95": f"{fmt_int(env.isl_eff_mean)} / {fmt_int(env.isl_eff_p95)}",
            "TTFT p95 : frontal / place / attente prefill / prefill (ms)": (
                f"{ev.frontend_s * 1e3:.1f} / {ev.slot_wait_s * 1e3:.1f} / {ev.prefill_wait_p95_s * 1e3:.1f}"
                f" / {ev.own_prefill_s * 1e3:.1f}"),
            "Poids lus par GPU et par step (Go)": round(ev.loaded.weights_bytes / 1e9, 1),
            "KV lu par GPU et par step (Go)": round(ev.loaded.kv_bytes / 1e9, 1),
            **({"Frontal : processus API × threads du renderer": f"{ev.frontend.api_servers} × {ev.frontend.workers}",
                "Frontal : requêtes/s et sorties streamées/s": (f"{fmt_int(ev.frontend.requests_per_s)} /"
                                                               f" {fmt_int(ev.frontend.chunks_per_s)}"),
                "Frontal : GIL / pool du renderer par processus (%)": (f"{ev.frontend.gil_utilization * 100:.0f} /"
                                                                       f" {ev.frontend.pool_utilization * 100:.0f}"),
                "Frontal : cœurs CPU": round(ev.frontend.cores, 2)} if ev.frontend else {}),
            "Débit de sortie par instance (tokens/s)": round(ev.throughput(env.a) or 0),
        }
    notes.append(f"Méthode : {KB} §3.1 (régime stable du budget), somme des rooflines par opérateur de"
                 " src/vllm_configurator/perf.py, file M/G/c pour le prefill et percentiles calculés sur les distributions.")
    return SloAnalysis(verdict=verdict, summary=summary, checks=checks, bottlenecks=bottlenecks, recommendations=recos,
                       step_breakdown=breakdown, calibration=calibration, calibration_prefill=calibration_prefill,
                       uncertainty=env.spread if model is not None else None,
                       uncertainty_ttft=env.spread_prefill if model is not None else None, test_plan=plan, notes=notes)


def adjusted_kv_tokens(ctx: RuleContext, state: dict[str, Any]) -> tuple[int | None, str | None]:
    """KV tokens at the generated gpu-memory-utilization when the log was read at another one."""
    kv_tokens = state.get("kv_tokens")
    observed, util = ctx.request.observed_kv, state.get("gpu_memory_utilization")
    weights = ctx.model.weights.checkpoint_bytes
    if not (kv_tokens and observed and observed.gpu_memory_utilization and util and weights):
        return kv_tokens, None
    if abs(observed.gpu_memory_utilization - util) < 1e-9:
        return kv_tokens, None
    tp, hbm = ctx.request.gpus_per_instance, ctx.gpu.memory.hbm_gb * 1e9
    free_obs = hbm * observed.gpu_memory_utilization - weights / tp
    free_cfg = hbm * util - weights / tp
    if free_obs <= 0 or free_cfg <= 0:
        return kv_tokens, None
    adjusted = int(kv_tokens * free_cfg / free_obs)
    return adjusted, (
        f"KV ramené à {fmt_int(adjusted)} tokens : mesuré à gpu-memory-utilization {observed.gpu_memory_utilization:g},"
        f" la config retient {util:g} (mémoire libre par GPU {free_obs / 1e9:.0f} → {free_cfg / 1e9:.0f} Go, estimation)."
    )


def _plan_point(env: Env, base: Point, seqs: int) -> Point:
    """Setting of one test step: the generated budget, raised only if the step needs it."""
    rho_max = env.ctx.heuristic("prefill_load_max").value
    budgets = sorted(set(BUDGET_GRID) | {base.budget})
    candidates = [b for b in budgets if b >= base.budget] or [base.budget]
    chosen = candidates[-1]
    for budget in candidates:
        if env.dspark_cap and seqs > budget // (env.k + 1):
            continue
        if evaluate(env, Point(seqs, budget, budget // 4, 1), inflight=seqs).rho <= rho_max:
            chosen = budget
            break
    threshold = base.threshold if chosen == base.budget else chosen // 4
    return Point(max_num_seqs=seqs, budget=chosen, threshold=threshold, instances=1)


def _pair(low: float | None, high: float | None, digits: int = 1) -> tuple[float, float] | None:
    if low is None or high is None:
        return None
    return (round(low, digits), round(high, digits) if not math.isinf(high) else math.inf)


def build_test_plan(env: Env, base: Point) -> TestPlan:
    """Load steps to validate in order; the last validated one sizes the deployment."""
    ctx, wl, slo = env.ctx, env.wl, env.ctx.request.slo
    tp = ctx.request.gpus_per_instance
    target = wl.target_concurrency
    per_instance = math.ceil(target / base.instances) if target else None
    cap = kv_seq_cap(env)
    top = max(base.max_num_seqs, per_instance or 0)
    if cap:
        top = min(top, cap)
    ladder = [s for s in ctx.heuristic("test_plan_ladder").value if s <= top]
    seqs = sorted(set(ladder) | {min(base.max_num_seqs, top)} | ({min(per_instance, top)} if per_instance else set()))

    rows = []
    for count in seqs:
        pt = _plan_point(env, base, count)
        ev, lo, hi = evaluate_range(env, pt, inflight=count)
        checks = [c for c in checks_for(env, ev) if c.in_verdict and c.key != "slots"]
        status = worst(checks)
        if env.model is None and status == "ok":
            status = "unknown"  # capacité suffisante, mais latences non estimées : la mesure tranche
        rows.append((count, pt, ev, lo, hi, checks, status))

    generated = min(base.max_num_seqs, top)
    if env.model is None:
        keep = rows if len(rows) <= 8 else rows[-8:]
    else:
        bad = next((i for i, r in enumerate(rows) if r[6] in ("at_risk", "ko")), None)
        if bad is None:
            keep = rows[-5:]
        else:
            ko = next((i for i in range(bad, len(rows)) if rows[i][6] == "ko"), len(rows) - 1)
            keep = rows[max(0, bad - 2):ko + 1]
            if len(keep) > 8:
                keep = keep[:3] + keep[3:-2][:: max(1, (len(keep) - 5) // 3)] + keep[-2:]
    if not any(r[0] == generated for r in keep):
        keep = sorted(keep + [r for r in rows if r[0] == generated], key=lambda r: r[0])

    steps = []
    for index, (count, pt, ev, lo, hi, checks, status) in enumerate(keep, 1):
        reasons = []
        for c in checks:
            if c.key in ("itl", "ttft") and c.estimate_low is not None:
                reasons.append(f"{c.label} {_range_text(c.estimate_low, c.estimate_high)}"
                               + (f" (cible {_ms(c.target)})" if c.target else ""))
            elif c.status in ("at_risk", "ko"):
                reasons.append(f"{c.label} {c.estimate:g} {c.unit or ''}".strip())
        if env.model is None and status == "unknown":
            reasons.append("KV et prefill suffisants ; latences non estimées")
        instances = math.ceil(target / count) if target else None
        steps.append(TestStep(
            index=index, max_num_seqs=count, max_num_batched_tokens=pt.budget, long_prefill_token_threshold=pt.threshold,
            capture_size=f.clamp_capture_size(f.cudagraph_capture_size(env.k, count), pt.budget, env.k + 1),
            inflight_per_instance=count,
            itl_ms=_pair(lo.itl_ms, hi.itl_ms), tpot_ms=_pair(lo.tpot_ms, hi.tpot_ms),
            ttft_ms=_pair(lo.ttft_ms, hi.ttft_ms),
            output_tokens_per_s=_pair(hi.throughput(env.a), lo.throughput(env.a), 0),
            kv_usage_pct=None if ev.kv_usage is None else round(ev.kv_usage * 100, 1),
            prefill_load_pct=None if math.isinf(ev.rho) else round(ev.rho * 100, 1),
            expectation=status, expectation_reason=" ; ".join(reasons),
            instances_needed=instances, gpus_needed=None if instances is None else instances * tp,
            is_generated_config=count == generated and pt.budget == base.budget,
        ))

    safe = max((s.max_num_seqs for s in steps if s.expectation == "ok"), default=None)
    possible = max((s.max_num_seqs for s in steps if s.expectation in ("ok", "at_risk")), default=None)
    if env.model is None:
        summary = (f"{len(steps)} paliers de {steps[0].max_num_seqs} à {steps[-1].max_num_seqs} séquences par instance :"
                   " sans modèle de latence, la limite n'est pas estimée, la mesure tranche.")
    elif possible is None:
        summary = ("Même le plus petit palier risque de manquer les SLO d'après l'estimation : commencer par lui pour mesurer"
                   " le plancher de latence, puis voir les pistes d'architecture.")
    else:
        if safe == possible:
            summary = (f"Tous les paliers jusqu'à {possible} séquences par instance devraient passer, même dans le cas"
                       f" pessimiste (estimation ×{env.spread[0]:g} à ×{env.spread[1]:g}).")
        else:
            summary = (f"Limite attendue entre {safe or steps[0].max_num_seqs} et {possible} séquences par instance"
                       f" (estimation ×{env.spread[0]:g} à ×{env.spread[1]:g}).")
        if target:
            low_n, high_n = math.ceil(target / possible), math.ceil(target / (safe or steps[0].max_num_seqs))
            summary += (f" Pour {fmt_int(target)} requêtes en vol : {_instances(low_n)}" if low_n == high_n else
                        f" Pour {fmt_int(target)} requêtes en vol : entre {low_n} et {high_n} instances") + f" de TP{tp}."

    duration_s = None
    if env.model is not None:
        ev_base = evaluate(with_factor(env, 1), base)
        if ev_base.average is not None:
            duration_s = env.osl_mean / env.a * ev_base.average.seconds + (ev_base.ttft_ms or 0) / 1e3
    soak = max(10.0, 3 * duration_s / 60) if duration_s else 10.0
    gen_metric = ("ITL p95 (vllm:inter_token_latency_seconds)" if env.metric == "itl"
                  else "TPOT p95 (vllm:request_time_per_output_token_seconds)")
    protocol = [
        "Déployer une instance avec les args générés, en ne changeant que --max-num-seqs, --max-num-batched-tokens,"
        " --long-prefill-token-threshold et la capture des CUDA graphs du palier.",
        "Injecter en boucle fermée, par rejeu de traces réelles (mêmes ISL, OSL et thinking qu'en production), autant de"
        " requêtes en vol que max-num-seqs : chaque requête terminée est aussitôt remplacée"
        " (par exemple vllm bench serve --max-concurrency sur un jeu de traces). Rejouer des sessions entières, dans"
        " l'ordre et avec le même routage qu'en production : des prompts aléatoires ou mélangés n'ont pas le hit rate de"
        " la production et font exploser le TTFT.",
        "Garder la dispersion réelle des sorties : une OSL fixe (ignore_eos avec une longueur constante) et une montée en"
        " charge d'un bloc font finir les requêtes par vagues, et leurs prefills arrivent en rafale (TTFT p95 doublé).",
        f"Mesurer en régime stable pendant au moins {soak:.0f} min après la montée en charge"
        + (f" (une requête dure jusqu'à ~{duration_s / 60:.1f} min ici)." if duration_s else "."),
        f"Relever {gen_metric}, TTFT p95 (vllm:time_to_first_token_seconds), vllm:num_requests_waiting,"
        " vllm:num_preemptions, vllm:kv_cache_usage_perc et le hit rate (vllm:prefix_cache_hits /"
        " vllm:prefix_cache_queries). Côté client, le TTFT se compte au premier token émis, raisonnement compris : avec"
        " le thinking, le premier token de « content » arrive après tout le raisonnement.",
        "Surveiller le CPU de chaque processus API (top -H, pidstat ; py-spy dump pour voir où il passe son temps) :"
        " un processus à ~100 % d'un cœur, et un TTFT ou une ITL client qui s'écartent de plus en plus du serveur quand"
        " la charge monte, signalent un frontal saturé (--api-server-count, --renderer-num-workers).",
        "Palier validé : passer au suivant. Premier palier en échec : la capacité d'une instance est le palier précédent.",
        "Saisir la latence de génération et le TTFT mesurés au palier de la config générée (★) dans « Mesures en"
        " production » : l'app recale séparément la partie decode et la partie prefill du modèle et resserre les plages"
        " des autres paliers.",
        "Dimensionner : instances = ⌈requêtes en vol visées / max-num-seqs validé⌉ (colonne « Si validé »), plus une marge"
        " pour les pics.",
    ]
    criteria = []
    if slo.itl_p95_ms:
        criteria.append(f"{gen_metric} ≤ {_ms(slo.itl_p95_ms)}")
    if slo.ttft_p95_ms:
        criteria.append(f"TTFT p95 ≤ {_ms(slo.ttft_p95_ms)}, sans file d'attente (vllm:num_requests_waiting ≈ 0)")
    criteria += [
        "vllm:num_preemptions stable (pas de préemption) et vllm:kv_cache_usage_perc < 90 %",
        "Démarrage sans OOM pendant la capture des CUDA graphs",
    ]
    notes = []
    if target:
        notes.append(
            "Requêtes en vol ≠ utilisateurs : un utilisateur agentique n'a une requête en cours qu'une partie du temps"
            " (outils, lecture, pauses). Mesurer vllm:num_requests_running + vllm:num_requests_waiting sur tous les pods,"
            " ou débit de requêtes × durée d'une requête" + (f" (~{duration_s:.0f} s ici)." if duration_s else ".")
        )
    if env.k == 0 and ctx.model.speculative.native_method:
        notes.append(
            f"Spec decoding désactivé : avec {ctx.model.speculative.native_method}, chaque requête finit ~a fois plus vite,"
            " donc moins de requêtes en vol pour le même trafic. L'ITL (un step) augmente un peu, le TPOT baisse nettement :"
            " comparer les deux variantes sur un même palier."
        )
    return TestPlan(steps=steps, summary=summary, protocol=protocol, pass_criteria=criteria, notes=notes)


MAX_TP = 8  # domaine NVLink d'un nœud HGX


def _wider_tp(env: Env, base: Point) -> tuple[int, int, Point, Evaluation] | None:
    """Same search with twice the tensor parallel, starting from the same number of GPUs."""
    tp2 = env.model.tp * 2
    if tp2 > MAX_TP:
        return None
    env2 = replace(env, model=replace(env.model, tp=tp2))
    start = max(1, base.instances * env.model.tp // tp2)
    base2 = replace(base, instances=start)
    found = best_point(env2, candidate_points(env2, base2, start), base2)
    if found is not None:
        return start, tp2, found[0], found[1]
    scaled = _scale_out(env2, base2)
    return None if scaled is None else (scaled[0], tp2, scaled[1], scaled[2])


def _scale_out(env: Env, base: Point) -> tuple[int, Point, Evaluation] | None:
    """Smallest number of instances with a passing operating point (exponential search, then bisection)."""
    def attempt(n: int):
        return best_point(env, candidate_points(env, base, n), base)

    limit = max(base.instances + 1, env.wl.target_concurrency)  # au-delà : moins d'une séquence par instance
    lo, hi, found = base.instances, base.instances + 1, None
    while True:
        found = attempt(hi)
        if found is not None:
            break
        if hi >= limit:
            return None
        lo, hi = hi, min(hi * 2, limit)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        candidate = attempt(mid)
        if candidate is not None:
            hi, found = mid, candidate
        else:
            lo = mid
    return hi, found[0], found[1]


def _instances(n: int) -> str:
    return f"{n} instance" + ("s" if n > 1 else "")


def _gpus(total: int, now: int) -> str:
    return f"{total} GPU, autant qu'aujourd'hui" if total == now else f"{total} GPU, contre {now} aujourd'hui"


def _point_summary(env: Env, ev: Evaluation) -> str:
    parts = [f"{ev.active} séquences actives par instance"
             + (f" ({ev.per_instance - ev.active} en attente d'une place)" if ev.per_instance and ev.per_instance > ev.active else "")]
    if ev.itl_ms is not None:
        parts.append(f"ITL p95 ≈ {_ms(ev.itl_ms)}")
    if ev.ttft_ms is not None:
        parts.append(f"TTFT p95 ≈ {_ms(ev.ttft_ms)}")
    if not math.isinf(ev.rho):
        parts.append(f"charge de prefill {ev.rho:.0%}")
    if ev.kv_usage is not None:
        parts.append(f"KV {ev.kv_usage:.0%}")
    return "Estimation : " + ", ".join(parts) + "."


def _summary(verdict: CheckStatus, bottlenecks: list[Bottleneck], slo_given: bool, has_model: bool) -> str:
    if verdict == "ok":
        head = "Les SLO devraient être tenus." if slo_given else "La capacité suffit (aucun SLO de latence fourni)."
        return head + ("" if has_model else " Latences non estimées faute de données de performance au catalogue.")
    head = {"at_risk": "SLO à risque", "ko": "SLO non tenus", "unknown": "Verdict impossible"}[verdict]
    if not bottlenecks:
        return head + "."
    return f"{head} : goulot principal — {bottlenecks[0].label.lower()}."
