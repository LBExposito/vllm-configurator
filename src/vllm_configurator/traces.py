"""working-set agent traces (`ws trace pi|claude`, https://github.com/T0mSIlver/working-set): the workload of the
configurator, read from recorded sessions instead of guessed.

A trace holds, per session and per model request, only sizes and waits (format ``workingset-agent-trace``,
versions 1 and 2)::

    {"format": "workingset-agent-trace", "version": 2, "source": "claude",
     "sessions": [{"requests": [{"prompt_tokens": 120000, "output_tokens": 800, "gap_s": null, "after": "start",
                                 "kind": "start", "service_s": 21.3}, ...]}, ...]}

What is derived, request by request, under a perfect prefix cache (no eviction, every request routed where its
session lives) — the same accounting as `ws replay`:

    cold   a session's first request, or a prompt that shrank (compaction, branch): everything past the prefix
           shared by all sessions is prefilled
    warm   the previous prompt is cached (and the previous reply when ``reply_cached``): only the growth is
           prefilled (tool outputs, the person's message)

The hit rate and the cold share read off a trace are therefore floors on what production sees: evictions and
misrouting only add cold requests. The waits give the share of time a session has a request in flight, which turns
a number of sessions into requests in flight (running) and paused sessions (whose KV must stay cached).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

FORMAT = "workingset-agent-trace"
VERSIONS = (1, 2)
# Waits where the session was not in use (error and retry, resume, session left open): `ws replay` replaces them with a
# human wait drawn from the trace (« sample » policy, the default).
INACTIVE = frozenset({"error", "resume", "idle"})
# Beyond this, a human wait is a session left open (same threshold as working-set).
PARK_S = 1800.0
# Bounds of the dispersion derived from the percentiles of the warm requests.
SIGMA_BOUNDS = (0.1, 3.0)


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class TraceProfile:
    sessions: int
    requests: int
    isl: dict[str, int]
    osl: dict[str, int]
    first_prompt_p50: int
    first_prompt_min: int
    max_context: int
    hit_rate: float                     # token-weighted, perfect cache
    cold_fraction: float                # share of cold requests
    cold_share_of_prefill: float        # share of the uncached prefill carried by cold requests
    warm_uncached: dict[str, int] | None
    mean_gap_s: float | None
    mean_service_s: float | None
    busy_share: float | None            # share of the time a session has a request in flight (recorded server)
    resets: int                         # prompts that shrank (compaction, branch)
    shared_prefix_tokens: int
    reply_cached: bool
    notes: list[str] = field(default_factory=list)

    def workload_updates(self, sessions: int | None = None) -> dict[str, Any]:
        """Workload fields read off the trace. With a number of sessions (all instances) and service times in the
        trace, also the requests in flight and the paused sessions."""
        updates: dict[str, Any] = {
            "isl": self.isl, "osl": self.osl,
            "prefix_cache_hit_rate": round(self.hit_rate, 4),
            "cold_request_fraction": round(self.cold_fraction, 4),
        }
        if self.warm_uncached:
            updates["warm_uncached_tokens"] = self.warm_uncached
        if self.mean_gap_s is not None:
            updates["session_gap_s"] = round(self.mean_gap_s, 1)
        if sessions and self.busy_share is not None:
            inflight = max(1, min(sessions, round(sessions * self.busy_share)))
            updates["target_concurrency"] = inflight
            updates["paused_sessions"] = sessions - inflight
        return updates


def parse_trace(data: str | bytes | dict) -> dict:
    """Validate a working-set trace (JSON text or parsed) and return it."""
    if not isinstance(data, dict):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TraceError(f"JSON invalide : {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise TraceError(f"Pas une trace working-set (format attendu {FORMAT!r}, produit par `ws trace pi|claude`).")
    if data.get("version") not in VERSIONS:
        raise TraceError(f"Version de trace {data.get('version')!r} non lue (versions lues : 1 et 2).")
    sessions = data.get("sessions")
    if not isinstance(sessions, list) or not any(isinstance(s, dict) and s.get("requests") for s in sessions):
        raise TraceError("La trace ne contient aucune requête.")
    for i, s in enumerate(sessions):
        for j, r in enumerate(s.get("requests", [])):
            try:
                if int(r["prompt_tokens"]) <= 0 or int(r["output_tokens"]) < 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as exc:
                raise TraceError(f"Session {i}, requête {j} : prompt_tokens > 0 et output_tokens ≥ 0 attendus.") from exc
    return data


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of a sorted list."""
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _percentiles(values: list[float], with_p99: bool = True) -> dict[str, int]:
    xs = sorted(values)
    out = {"p50": max(1, round(_quantile(xs, 0.5))), "p95": max(1, round(_quantile(xs, 0.95)))}
    if with_p99:
        out["p99"] = max(1, round(_quantile(xs, 0.99)))
    # rounding: keep the order Percentiles requires
    out["p95"] = max(out["p95"], out["p50"])
    if with_p99:
        out["p99"] = max(out["p99"], out["p95"])
    return out


def warm_sigma(p50: float, p95: float) -> float:
    """Dispersion of a lognormal with these median and 95th percentile, within SIGMA_BOUNDS."""
    sigma = math.log(max(p95, p50) / p50) / 1.6449 if p50 > 0 else SIGMA_BOUNDS[1]
    return min(max(sigma, SIGMA_BOUNDS[0]), SIGMA_BOUNDS[1])


def _mean_gap(requests: list[dict]) -> tuple[float | None, int]:
    """Mean wait between two requests of a session as `ws replay` sends them (policy « sample »): inactive gaps
    replaced by the mean human wait of the trace, the others capped at PARK_S. Returns (mean, dropped inactive)."""
    think = [r["gap_s"] for r in requests if r.get("kind") == "think" and r.get("gap_s") is not None
             and r["gap_s"] <= PARK_S]
    sampled = sum(think) / len(think) if think else None
    gaps, dropped = [], 0
    for r in requests:
        gap = r.get("gap_s")
        if gap is None:
            continue
        if r.get("kind") in INACTIVE:
            if sampled is None:
                dropped += 1
                continue
            gaps.append(sampled)
        else:
            gaps.append(min(float(gap), PARK_S))
    return (sum(gaps) / len(gaps) if gaps else None), dropped


def profile_trace(data: str | bytes | dict, shared_prefix_tokens: int = 0, reply_cached: bool = True) -> TraceProfile:
    """Workload of a working-set trace (see the module docstring)."""
    doc = parse_trace(data)
    sessions = [s["requests"] for s in doc["sessions"] if isinstance(s, dict) and s.get("requests")]
    shared = max(0, int(shared_prefix_tokens))
    prompts, outputs, firsts = [], [], []
    total_prompt = total_u = cold_u = 0.0
    cold = resets = 0
    warm_u: list[float] = []
    max_context = 0
    for reqs in sessions:
        prev: tuple[int, int] | None = None
        for i, r in enumerate(reqs):
            p_tok, o_tok = int(r["prompt_tokens"]), int(r["output_tokens"])
            prompts.append(p_tok)
            outputs.append(max(o_tok, 1))
            max_context = max(max_context, p_tok + o_tok)
            if i == 0:
                firsts.append(p_tok)
            if prev is None or p_tok < prev[0]:
                if prev is not None:
                    resets += 1
                cold += 1
                u = max(p_tok - min(shared, p_tok), 1)
                cold_u += u
            else:
                cached = min(prev[0] + (prev[1] if reply_cached else 0), p_tok)
                u = p_tok - cached
                warm_u.append(u)
            total_prompt += p_tok
            total_u += u
            prev = (p_tok, o_tok)
    n = len(prompts)
    positive = [u for u in warm_u if u > 0]
    warm = _percentiles(positive, with_p99=False) if len(positive) >= 2 else None

    all_reqs = [r for reqs in sessions for r in reqs]
    gap, dropped = _mean_gap(all_reqs)
    services = [float(r["service_s"]) for r in all_reqs if r.get("service_s") is not None]
    service = sum(services) / len(services) if services else None
    busy = service / (service + gap) if service is not None and gap is not None and service + gap > 0 else None

    notes = []
    ws_default = min(firsts)
    if ws_default > max(shared, 1) * 2 and ws_default > 4096:
        notes.append(
            f"`ws replay` prend par défaut le plus petit premier prompt de la trace ({_tok(ws_default)} tokens) comme préfixe"
            f" commun à toutes les sessions : leurs premiers tours deviennent quasi chauds. Si ce contexte est propre à"
            f" chaque session en production, passer `--shared-prefix-tokens {shared}` (le préfixe commun réel :"
            " system prompt + outils), sinon le test ne voit presque aucun prefill froid."
        )
    if not reply_cached:
        notes.append("Réponses non réutilisées : le chat template retire la réponse précédente (raisonnement) de"
                     " l'historique, donc elle est prefillée à nouveau au tour suivant.")
    if dropped:
        notes.append(f"{dropped} attentes inactives (erreur, reprise, session laissée ouverte) ignorées : la trace n'a"
                     " pas d'attente humaine pour les remplacer.")
    if service is None:
        notes.append("Trace sans durées de service (version 1) : la part du temps en vol n'est pas connue, saisir les"
                     " requêtes en vol et les sessions en pause à la main.")
    else:
        notes.append(f"Durée de service enregistrée : {service:.1f} s par requête en moyenne, sur le serveur de"
                     " l'enregistrement. La part en vol qui en découle est indicative : le plan de test la recalcule avec"
                     " la durée estimée sur cette config.")
    notes.append("Hit rate et requêtes froides d'un cache parfait (ni éviction ni mauvais routage) : planchers de ce que"
                 " verra la production. Comparer au hit rate mesuré (vllm:prefix_cache_hits / vllm:prefix_cache_queries)"
                 " et à la part d'évictions que rapporte `ws replay`.")
    return TraceProfile(
        sessions=len(sessions), requests=n, isl=_percentiles(prompts), osl=_percentiles(outputs),
        first_prompt_p50=round(_quantile(sorted(firsts), 0.5)), first_prompt_min=ws_default, max_context=max_context,
        hit_rate=1.0 - total_u / total_prompt if total_prompt else 0.0, cold_fraction=cold / n if n else 0.0,
        cold_share_of_prefill=cold_u / total_u if total_u else 0.0, warm_uncached=warm, mean_gap_s=gap,
        mean_service_s=service, busy_share=busy, resets=resets, shared_prefix_tokens=shared,
        reply_cached=reply_cached, notes=notes,
    )


def _tok(x: float) -> str:
    return f"{round(x):,}".replace(",", " ")


def summary_rows(profile: TraceProfile) -> list[tuple[str, str]]:
    """(label, value) rows describing what was read off the trace."""
    tok = _tok
    rows = [
        ("Sessions / requêtes", f"{profile.sessions} / {profile.requests}"
                                f" ({profile.requests / max(profile.sessions, 1):.1f} requêtes par session)"),
        ("ISL p50 / p95 / p99", " / ".join(tok(profile.isl[k]) for k in ("p50", "p95", "p99"))),
        ("Premier prompt p50 (min)", f"{tok(profile.first_prompt_p50)} ({tok(profile.first_prompt_min)})"),
        ("OSL p50 / p95 / p99", " / ".join(tok(profile.osl[k]) for k in ("p50", "p95", "p99"))),
        ("Contexte max (prompt + sortie)", tok(profile.max_context)),
        ("Hit rate (cache parfait, pondéré tokens)", f"{profile.hit_rate:.1%}"),
        ("Requêtes froides", f"{profile.cold_fraction:.1%} des requêtes, {profile.cold_share_of_prefill:.0%} du prefill"
                             f" non caché ({profile.resets} prompts rétrécis)"),
    ]
    if profile.warm_uncached:
        w = profile.warm_uncached
        rows.append(("Tokens à prefiller d'une requête chaude p50 / p95",
                     f"{tok(w['p50'])} / {tok(w['p95'])} (σ = {warm_sigma(w['p50'], w['p95']):.2f})"))
    if profile.mean_gap_s is not None:
        rows.append(("Attente moyenne entre deux requêtes", f"{profile.mean_gap_s:.1f} s"))
    if profile.busy_share is not None:
        rows.append(("Part du temps en vol (serveur enregistré)", f"{profile.busy_share:.0%}"))
    return rows


# --- Results of `ws replay --out run.json` --------------------------------------------------------------------------

RUN_FORMAT = "workingset-replay-run"
RUN_VERSIONS = (1,)
MIN_DECODE_TOKENS = 32  # same threshold as working-set for the decode speed of a request


@dataclass(frozen=True)
class RunStep:
    """One judged step of a replay: N sessions at once, what the requests of its window measured."""

    sessions: int
    passed: bool
    no_evidence: bool
    requests: int
    inflight: float | None              # requests in flight, mean over the window, whole deployment
    ttft_p95_ms: float | None           # every request of the window, cold ones included
    warm_ttft_p90_ms: float | None
    tpot_p95_ms: float | None           # (last token − first) / (tokens − 1), per request
    decode_p10_tok_s: float | None
    hit_rate: float | None              # tokens served from the cache / prompt tokens (vLLM usage)
    cold_fraction: float | None         # requests with nothing cached past the shared prefix
    eviction_share: float | None
    server_hit_rate: float | None
    kv_max: float | None
    preemptions: float | None
    waiting_max: float | None
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunProfile:
    steps: list[RunStep]
    serves: int
    fails_at: int | None
    endpoints: int
    routing: str
    shared_prefix_tokens: int
    kv_block_tokens: int
    complete: bool
    notes: list[str] = field(default_factory=list)

    def default_step(self) -> int:
        """Index of the step to recalibrate on: the largest that passed (closest to the limit), else the smallest."""
        passed = [i for i, s in enumerate(self.steps) if s.passed]
        if passed:
            return max(passed, key=lambda i: self.steps[i].sessions)
        return min(range(len(self.steps)), key=lambda i: self.steps[i].sessions)

    def updates(self, index: int, instances: int) -> dict[str, dict[str, Any]]:
        """Request fields read off one step: measured latencies and load (observed_perf), measured cache (workload).
        `instances`: vLLM instances behind the endpoint(s), to bring the load to one instance."""
        step = self.steps[index]
        n = max(instances, 1)
        perf: dict[str, Any] = {"sessions_per_instance": round(step.sessions / n, 2),
                                "capacity_sessions_per_instance": round(self.serves / n, 2)}
        if step.ttft_p95_ms:
            perf["ttft_p95_ms"] = round(step.ttft_p95_ms, 1)
        if step.tpot_p95_ms:
            perf["tpot_p95_ms"] = round(step.tpot_p95_ms, 2)
        if step.inflight:
            perf["inflight_per_instance"] = round(step.inflight / n, 2)
        workload: dict[str, Any] = {}
        hit = step.hit_rate if step.hit_rate is not None else step.server_hit_rate
        if hit is not None:
            workload["prefix_cache_hit_rate"] = round(hit, 4)
        if step.cold_fraction is not None:
            workload["cold_request_fraction"] = round(step.cold_fraction, 4)
        return {"observed_perf": perf, "workload": workload}


def _rows_in_window(step: dict) -> list[dict]:
    """Requests in flight at any time of the measured window (working-set's own rule)."""
    w0, w1 = (step.get("window") or (0.0, 0.0))[:2]
    end_all = w1 + (step.get("drain_s") or 0.0)
    out = []
    for r in step.get("rows", []):
        if r.get("sent") is None or r["sent"] >= w1:
            continue
        if r.get("done"):
            end = r.get("ended")
            if end is None:
                end = r["sent"] + (r.get("service") or 0.0)
        else:
            end = end_all
        if end >= w0:
            out.append(r)
    return out


def _ok(r: dict) -> bool:
    return bool(r.get("done")) and not r.get("error") and r.get("ttft") is not None


def _generated(r: dict) -> int:
    return int(r.get("ctok") or r.get("out_tokens") or 0)


def _run_step(step: dict, shared: int, block: int) -> RunStep:
    rows = _rows_in_window(step)
    ok = [r for r in rows if _ok(r)]
    ttfts = sorted(r["ttft"] * 1e3 for r in ok)
    warm = sorted(r["ttft"] * 1e3 for r in ok if r.get("warm"))
    tpots = sorted(r["span"] / (_generated(r) - 1) * 1e3 for r in ok
                   if _generated(r) >= MIN_DECODE_TOKENS and r.get("span") and r["span"] > 0)
    reported = [r for r in ok if r.get("cached") is not None and (r.get("ptok") or r.get("prompt"))]
    hit = cold = None
    if reported:
        prompt = sum(r.get("ptok") or r["prompt"] for r in reported)
        hit = sum(min(r["cached"], r.get("ptok") or r["prompt"]) for r in reported) / prompt if prompt else None
        cold = sum(1 for r in reported if r["cached"] < shared // block * block + block) / len(reported)
    checks = step.get("checks") or {}
    evictions = (checks.get("evictions") or {}).get("value")
    server = step.get("server") or {}
    return RunStep(
        sessions=int(step["sessions"]), passed=bool(step.get("passed")), no_evidence=bool(step.get("no_evidence")),
        requests=len(rows), inflight=step.get("streams_mean"),
        ttft_p95_ms=_quantile(ttfts, 0.95) if ttfts else None,
        warm_ttft_p90_ms=_quantile(warm, 0.90) if warm else None,
        tpot_p95_ms=_quantile(tpots, 0.95) if tpots else None,
        decode_p10_tok_s=1e3 / _quantile(tpots, 0.90) if tpots else None,
        hit_rate=hit, cold_fraction=cold, eviction_share=evictions,
        server_hit_rate=server.get("prefix_hit_rate"), kv_max=server.get("kv_max"),
        preemptions=server.get("preemptions"), waiting_max=server.get("waiting_max"),
        reasons=list(step.get("reasons") or []),
    )


def profile_run(data: str | bytes | dict) -> RunProfile:
    """Steps of a working-set replay record (`ws replay --out run.json`)."""
    if not isinstance(data, dict):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TraceError(f"JSON invalide : {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != RUN_FORMAT:
        raise TraceError(f"Pas un résultat de ws replay (format attendu {RUN_FORMAT!r}, produit par `ws replay --out`).")
    if data.get("version") not in RUN_VERSIONS:
        raise TraceError(f"Version de résultat {data.get('version')!r} non lue (version lue : 1).")
    plan = data.get("plan") or {}
    shared = int(plan.get("shared_prefix_tokens") or 0)
    block = max(int(plan.get("kv_block_tokens") or 16), 1)
    raw = [s for s in data.get("steps") or [] if isinstance(s, dict) and s.get("sessions")]
    if not raw:
        raise TraceError("Le résultat ne contient aucun palier jugé.")
    steps = sorted((_run_step(s, shared, block) for s in raw), key=lambda s: s.sessions)
    answer = data.get("answer") or {}
    endpoints = len(data.get("endpoints") or []) or 1
    notes = []
    if not data.get("complete", True):
        notes.append("Résultat incomplet (rejeu interrompu) : seuls les paliers déjà jugés sont lus.")
    if all(s.hit_rate is None for s in steps):
        notes.append("Le serveur n'a renvoyé aucun cached_tokens (vLLM sans --enable-prompt-tokens-details) : hit rate"
                     " et requêtes froides non mesurés par requête, évictions non comptées.")
    if all(s.server_hit_rate is None and s.kv_max is None for s in steps):
        notes.append("Rejeu sans /metrics (--no-metrics) : ni file d'attente, ni préemptions, ni occupation du KV.")
    notes.append(f"Préfixe commun du rejeu : {_tok(shared)} tokens (--shared-prefix-tokens). Le hit rate et la part froide"
                 " mesurés en dépendent : il doit correspondre au préfixe réellement commun en production.")
    if endpoints == 1:
        notes.append("Une seule URL : si c'est une passerelle (llm-d) devant plusieurs pods, la charge est ramenée à une"
                     " instance avec le champ « Instances », et le routage fait partie de la mesure.")
    return RunProfile(steps=steps, serves=int(answer.get("serves") or 0), fails_at=answer.get("fails_at"),
                      endpoints=endpoints, routing=str(plan.get("routing") or "sticky"), shared_prefix_tokens=shared,
                      kv_block_tokens=block, complete=bool(data.get("complete", True)), notes=notes)


def run_rows(profile: RunProfile) -> list[dict[str, Any]]:
    """One table row per step of a replay."""
    def ms(v: float | None) -> str:
        return "—" if v is None else f"{_tok(v)} ms"

    def pct(v: float | None) -> str:
        return "—" if v is None else f"{v:.1%}"
    return [{
        "Sessions": s.sessions,
        "Verdict": "✅" if s.passed else ("❔" if s.no_evidence else "🛑"),
        "Requêtes en vol (moy.)": "—" if s.inflight is None else f"{s.inflight:.1f}",
        "TTFT p95": ms(s.ttft_p95_ms), "TTFT chaud p90": ms(s.warm_ttft_p90_ms),
        "TPOT p95": "—" if s.tpot_p95_ms is None else f"{s.tpot_p95_ms:.1f} ms",
        "Hit rate": pct(s.hit_rate if s.hit_rate is not None else s.server_hit_rate),
        "Froides": pct(s.cold_fraction), "Évictions": pct(s.eviction_share),
        "KV max": pct(s.kv_max), "Pourquoi": " ; ".join(s.reasons),
    } for s in profile.steps]
