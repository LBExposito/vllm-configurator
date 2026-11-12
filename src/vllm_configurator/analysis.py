"""SLO analysis: will the generated configuration meet the targets, what is the bottleneck, what to change.

The analysis evaluates an operating point (max_num_seqs, token budget, long prefill threshold,
number of instances) with two kinds of checks:

- structural checks, which need no hardware model: KV occupancy, prefill load of the budget,
  queueing for a sequence slot;
- latency checks (ITL p95, TTFT p95), from the roofline step model of perf.py, recalibrated by
  the measured ITL when provided.

It then classifies the bottleneck and searches the parameter grid (and then the number of
instances) for an operating point that meets every target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from . import formulas as f
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
    # Incertitude multiplicative sur les durées estimées (bas, haut).
    spread: tuple[float, float] = (1.0, 1.0)

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

    @property
    def isl_eff_mean(self) -> float:
        return self.isl_mean * (1 - self.hit)

    @property
    def metric(self) -> str:
        return self.ctx.request.slo.itl_metric

    @property
    def step_target_ms(self) -> float | None:
        """Generation-latency target expressed as a step duration (TPOT × a with spec decoding)."""
        target = self.ctx.request.slo.itl_p95_ms
        if target is None:
            return None
        return target * self.a if self.metric == "tpot" else target

    @property
    def isl_eff_p95(self) -> float:
        return (self.wl.isl.p95 or self.wl.isl.p50) * (1 - self.hit)

    @property
    def isl_cold(self) -> int:
        return self.wl.isl.p95 or self.wl.isl.p50


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
    prefill_wait_s: float = 0.0
    own_prefill_s: float = 0.0
    cold_prefill_s: float = 0.0

    @property
    def itl_ms(self) -> float | None:
        return self.loaded.ms if self.loaded else None

    @property
    def tpot_ms(self) -> float | None:
        return None if self.average is None else self.average.ms

    @property
    def ttft_ms(self) -> float | None:
        if self.chunk_step is None:
            return None
        return (self.slot_wait_s + self.prefill_wait_s + self.own_prefill_s) * 1e3

    @property
    def ttft_cold_ms(self) -> float | None:
        if self.chunk_step is None:
            return None
        return (self.slot_wait_s + self.prefill_wait_s + self.cold_prefill_s) * 1e3

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
    m = p.PerfModel(
        tp=ctx.request.gpus_per_instance,
        hbm_bytes_per_s=gpu.bandwidth.hbm_tb_s * 1e12 * h("perf_hbm_efficiency"),
        flops_per_s=gpu.compute.dense_tflops[perf.compute_dtype] * 1e12 * h("perf_mfu"),
        overhead_s=arch.num_hidden_layers * h("perf_overhead_per_layer_ms") * 1e-3,
        active_params_decode=perf.active_params_per_output_token,
        active_params_prefill=perf.active_params_per_prompt_token or perf.active_params_per_output_token,
        dense_params=dense, dense_bytes_per_param=perf.dense_bytes_per_param,
        routed_params=routed, routed_bytes_per_param=routed_bpp,
        experts_per_tok=experts_per_tok, n_experts=n_experts, kv_bytes_per_token_per_gpu=kv_bpt,
    )
    notes = [
        f"Modèle de latence : {gpu.display_name} à {gpu.bandwidth.hbm_tb_s:g} To/s × {h('perf_hbm_efficiency'):g} et"
        f" {fmt_int(gpu.compute.dense_tflops[perf.compute_dtype])} TFLOPS {perf.compute_dtype} × {h('perf_mfu'):g},"
        + f" surcoût fixe {m.overhead_s * 1e3:.1f} ms par step ({arch.num_hidden_layers} couches × {h('perf_overhead_per_layer_ms'):g} ms)."
        f" Poids lus par step : {dense / 1e9:.1f} G paramètres non routés"
        + (f" + la part des {routed / 1e9:.0f} G paramètres d'experts routés touchée par au moins un token." if routed else ".")
        + f" KV : {kv_note}.",
    ]
    if ctx.model.kv_cache.bytes_per_token is not None and any(
            kind in ("sparse", "compressed_sparse", "sparse_dsa", "indexed_sparse") for kind in arch.attention):
        notes.append("Attention sparse : le modèle suppose la lecture de tout le contexte en decode (borne haute).")
    return m, notes


# --- Evaluation of one operating point ---------------------------------------

def evaluate(env: Env, pt: Point, inflight: int | None = None) -> Evaluation:
    """Evaluate an operating point. `inflight` forces the in-flight requests per instance (test steps)."""
    wl = env.wl
    if inflight is not None:
        per_instance = inflight
    else:
        per_instance = math.ceil(wl.target_concurrency / pt.instances) if wl.target_concurrency else None
    active = min(pt.max_num_seqs, per_instance) if per_instance else pt.max_num_seqs
    capacity = pt.budget - active * env.d
    # Régime stable : S·a/OSL requêtes finissent par step et sont remplacées (§3.1), longueurs moyennes.
    arrivals = active * env.a / env.osl_mean
    demand = arrivals * env.isl_eff_mean
    rho = math.inf if capacity <= 0 else demand / capacity
    kv_usage = active * env.context_tokens / env.kv_tokens if env.kv_tokens else None
    ev = Evaluation(point=pt, per_instance=per_instance, active=active, capacity=capacity, rho=rho,
                    kv_usage=kv_usage, slot_wait_s=0.0)
    m = env.model
    if m is None or capacity <= 0:
        return ev

    tokens_per_seq = env.k + 1
    ctx_tokens = env.context_tokens
    chunk = capacity if active <= 1 else min(pt.threshold, capacity)
    decode = p.step_time(m, active, tokens_per_seq, ctx_tokens, 0)
    average = p.step_time(m, active, tokens_per_seq, ctx_tokens, min(rho, 1.0) * capacity)

    # Part des steps qui portent du prefill ; au-delà de 5 %, l'ITL p95 est celle d'un step chargé.
    steps_per_prefill = math.ceil(env.isl_eff_mean / chunk) if env.isl_eff_mean >= 1 else 0
    share = min(1.0, arrivals * steps_per_prefill)
    if share >= 0.05:
        in_step = rho * capacity / share if share else 0.0
        p95_prefill = min(capacity, max(min(env.isl_eff_p95, chunk), in_step))
    else:
        p95_prefill = 0.0
    loaded = p.step_time(m, active, tokens_per_seq, ctx_tokens, p95_prefill)

    chunk_step = p.step_time(m, active, tokens_per_seq, ctx_tokens, min(capacity, max(chunk, min(rho, 1.0) * capacity)))
    slot_wait = 0.0
    if per_instance and per_instance > pt.max_num_seqs:
        duration = env.osl_mean / env.a * average.seconds
        slot_wait = (per_instance - pt.max_num_seqs) / pt.max_num_seqs * duration
    service = env.isl_eff_mean / capacity * average.seconds
    prefill_wait = p.md1_wait(rho, service)
    own = max(1, math.ceil(env.isl_eff_p95 / chunk)) * chunk_step.seconds
    cold = max(1, math.ceil(env.isl_cold / chunk)) * chunk_step.seconds
    return replace(ev, decode=decode, loaded=loaded, average=average, chunk=chunk, chunk_step=chunk_step,
                   slot_wait_s=slot_wait, prefill_wait_s=prefill_wait, own_prefill_s=own, cold_prefill_s=cold)


def with_factor(env: Env, factor: float) -> Env:
    if env.model is None or factor == 1.0:
        return env
    return replace(env, model=env.model.calibrated(env.model.calibration * factor))


def evaluate_range(env: Env, pt: Point, inflight: int | None = None) -> tuple[Evaluation, Evaluation, Evaluation]:
    """(central, optimistic, pessimistic) evaluations under the model uncertainty."""
    return (evaluate(env, pt, inflight), evaluate(with_factor(env, env.spread[0]), pt, inflight),
            evaluate(with_factor(env, env.spread[1]), pt, inflight))


def gen_latency_ms(env: Env, ev: Evaluation) -> float | None:
    """The generation latency the SLO is about: ITL (one loaded step) or TPOT (mean step / a)."""
    if ev.loaded is None:
        return None
    return ev.loaded.ms if env.metric == "itl" else ev.average.ms / env.a


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
    lo = evaluate(with_factor(env, env.spread[0]), ev.point, inflight) if env.model else ev
    hi = evaluate(with_factor(env, env.spread[1]), ev.point, inflight) if env.model else ev
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
            + (f" ITL = un step ; TPOT = step moyen / a ≈ {_ms(ev.average.ms / env.a)} avec a = {env.a:g}." if env.k else "")
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
            f"Plage {_range_text(lo.ttft_ms, hi.ttft_ms)}. Attente d'une place {_ms(ev.slot_wait_s * 1e3)}"
            f" + attente du prefill {_ms(ev.prefill_wait_s * 1e3)}"
            f" + prefill de {fmt_int(env.isl_eff_p95)} tokens non cachés (ISL P95 × (1 − {env.hit:.3f}))"
            f" en morceaux de {fmt_int(ev.chunk)} : {_ms(ev.own_prefill_s * 1e3)}."
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
                detail=(f"Cas pire, hors verdict : préfixe évincé du cache (session reprise après une longue pause),"
                        f" {fmt_int(env.isl_cold)} tokens à recalculer en {math.ceil(env.isl_cold / ev.chunk)} morceaux."),
            ))
    return out


def worst(checks: list[SloCheck]) -> CheckStatus:
    statuses = [c.status for c in checks if c.in_verdict and c.status != "unknown"]
    return max(statuses, key=RANK.get) if statuses else "unknown"


def passes(env: Env, ev: Evaluation) -> bool:
    return worst(checks_for(env, ev)) == "ok"


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
    for pt in points:
        ev = evaluate(env, pt)
        if not passes(env, ev):
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

    if bad("prefill_load") or (bad("ttft") and ev.prefill_wait_s >= max(ev.slot_wait_s, ev.own_prefill_s)):
        bottlenecks.append(Bottleneck(
            key="prefill", label="Prefill (débit du budget de tokens)", status=by["prefill_load"].status if bad("prefill_load") else by["ttft"].status,
            explanation=(
                "Le prefill non caché arrive plus vite que le budget ne peut l'absorber : les requêtes attendent avant leur premier"
                " token et la file grandit." if ev.rho >= 1 else
                f"Le budget de prefill est chargé à {ev.rho:.0%} : l'attente avant prefill ({_ms(ev.prefill_wait_s * 1e3)})"
                " domine le TTFT."
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
        parts = {"slots": ev.slot_wait_s, "prefill": ev.prefill_wait_s, "chunks": ev.own_prefill_s}
        main = max(parts, key=parts.get)
        if main == "slots":
            bottlenecks.append(Bottleneck(
                key="slots", label="Places de séquence (max-num-seqs)", status=by["ttft"].status,
                explanation=(f"{ev.per_instance - base.max_num_seqs} requêtes par instance attendent une place :"
                             f" {_ms(ev.slot_wait_s * 1e3)} d'attente, l'essentiel du TTFT."),
            ))
        elif main == "chunks":
            bottlenecks.append(Bottleneck(
                key="prefill_chunks", label="Prefill d'une longue requête (seuil de long prefill)", status=by["ttft"].status,
                explanation=(f"Le prompt P95 ({fmt_int(env.isl_eff_p95)} tokens non cachés) est découpé en morceaux de"
                             f" {fmt_int(ev.chunk)} : {math.ceil(env.isl_eff_p95 / ev.chunk)} steps de {_ms(ev.chunk_step.ms)}."),
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
    bottlenecks.sort(key=lambda b: -RANK[b.status])
    return bottlenecks, recos


# --- Entry point ---------------------------------------------------------------

def analyze(ctx: RuleContext, state: dict[str, Any]) -> SloAnalysis:
    req = ctx.request
    model, notes = build_perf_model(ctx, state)
    kv_tokens, kv_note = adjusted_kv_tokens(ctx, state)
    if kv_note:
        notes.append(kv_note)
    uncertainty = ctx.heuristic("perf_uncertainty").value
    env = Env(
        ctx=ctx, k=state["num_speculative_tokens"], d=state["decode_tokens_per_seq"], a=state["acceptance_length"],
        hit=state["prefix_cache_hit_rate"], kv_tokens=kv_tokens, model=model,
        dspark_cap=bool(ctx.behavior("spec_decode.dspark_active_seq_cap").value) and state.get("spec_method") in ("dspark", "dflash"),
        spread=tuple(uncertainty["uncalibrated"]),
    )
    base = Point(max_num_seqs=state["max_num_seqs"], budget=state["max_num_batched_tokens"],
                 threshold=state["long_prefill_token_threshold"], instances=req.num_instances)
    notes.append(
        "Convention vLLM (vllm/v1/metrics/stats.py) : l'ITL est l'intervalle entre deux sorties d'une requête, soit un step"
        " du scheduler, quel que soit le nombre de tokens de la sortie ; le TPOT divise la durée du decode par le nombre de"
        " tokens. Avec le speculative decoding, l'ITL augmente un peu mais le TPOT est divisé par ~a."
    )
    notes.append(
        f"Longueurs moyennes estimées d'après les percentiles : ISL {fmt_int(env.isl_mean)}, OSL {fmt_int(env.osl_mean)}"
        f" (P50 : {fmt_int(req.workload.isl.p50)} et {fmt_int(req.workload.osl.p50)})."
    )
    if not req.workload.target_concurrency:
        notes.append(f"Requêtes en vol non fournies : analyse à pleine charge ({base.max_num_seqs} séquences par instance).")

    calibration = None
    observed = req.observed_perf
    if model is not None and observed and observed.itl_p95_ms:
        raw = gen_latency_ms(env, evaluate(env, base))
        if raw:
            calibration = observed.itl_p95_ms / raw
            env = replace(env, model=model.calibrated(calibration), spread=tuple(uncertainty["calibrated"]))
            notes.append(
                f"Modèle recalibré sur la {'ITL' if env.metric == 'itl' else 'TPOT'} p95 mesurée ({observed.itl_p95_ms:g} ms"
                f" contre {raw:.1f} ms estimés) :"
                f" toutes les durées × {calibration:.2f}." + (" Écart important : vérifier que la mesure porte bien sur cette config."
                                                                 if not 0.5 <= calibration <= 2 else "")
            )

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
        tune = best_point(env, candidate_points(env, base, base.instances), base)
        if tune is not None:
            pt, tev = tune
            changes = changes_between(env, base, pt)
            if changes:
                recos.append(Recommendation(
                    kind="param", title="Réglage qui tient les SLO sans GPU supplémentaire : " + _describe(changes),
                    rationale=_point_summary(env, tev),
                    changes=changes, addresses=[b.key for b in bottlenecks],
                ))
        if req.workload.target_concurrency:
            tp = req.gpus_per_instance
            scale = _scale_out(env, base)
            wider = _wider_tp(env, base) if model is not None else None
            gpus_now = base.instances * tp
            if scale is not None:
                n, pt, sev = scale
                changes = {"instances": n, "GPU": n * tp} | changes_between(env, base, pt)
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
                               " instance (prudent) et même surcoût fixe par step (all-reduce plus large non modélisé)."),
                    changes={"instances": n2, "GPU": n2 * tp2, "--tensor-parallel-size": tp2}
                    | changes_between(env, base, pt2),
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
    if model is not None and not (observed and observed.itl_p95_ms):
        recos.append(Recommendation(
            kind="measure", title="Valider par paliers, puis saisir la latence mesurée pour recaler l'analyse",
            rationale=("Le modèle de latence est un ordre de grandeur (×"
                       f"{env.spread[0]:g} à ×{env.spread[1]:g}) : suivre le plan de paliers et saisir la latence mesurée"
                       " au palier de la config générée."),
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
            "Lecture mémoire / calcul / surcoût du step chargé (ms)": (
                f"{ev.loaded.memory_s * 1e3:.2f} / {ev.loaded.compute_s * 1e3:.2f} / {ev.loaded.overhead_s * 1e3:.2f}"),
            "Poids lus par GPU et par step (Go)": round(ev.loaded.weights_bytes / 1e9, 1),
            "KV lu par GPU et par step (Go)": round(ev.loaded.kv_bytes / 1e9, 1),
            "Débit de sortie par instance (tokens/s)": round(ev.throughput(env.a) or 0),
        }
    notes.append(f"Méthode : {KB} §3.1 (régime stable du budget) et modèle roofline de src/vllm_configurator/perf.py.")
    return SloAnalysis(verdict=verdict, summary=summary, checks=checks, bottlenecks=bottlenecks, recommendations=recos,
                       step_breakdown=breakdown, calibration=calibration,
                       uncertainty=env.spread if model is not None else None, test_plan=plan, notes=notes)


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
            itl_ms=_pair(lo.itl_ms, hi.itl_ms), tpot_ms=_pair(
                None if lo.average is None else lo.average.ms / env.a, None if hi.average is None else hi.average.ms / env.a),
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
        ev_base = evaluate(with_factor(env, env.spread[1]), base)
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
        " (par exemple vllm bench serve --max-concurrency sur un jeu de traces).",
        f"Mesurer en régime stable pendant au moins {soak:.0f} min après la montée en charge"
        + (f" (une requête dure jusqu'à ~{duration_s / 60:.1f} min ici)." if duration_s else "."),
        f"Relever {gen_metric}, TTFT p95 (vllm:time_to_first_token_seconds), vllm:num_requests_waiting,"
        " vllm:num_preemptions et vllm:kv_cache_usage_perc.",
        "Palier validé : passer au suivant. Premier palier en échec : la capacité d'une instance est le palier précédent.",
        "Saisir la latence mesurée au palier de la config générée (★) dans « Mesures en production » : l'app recale le"
        " modèle et resserre les plages des autres paliers.",
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
