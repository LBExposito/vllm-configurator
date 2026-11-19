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
# Attentes où la session n'était pas utilisée (erreur et nouvel essai, reprise, session laissée ouverte) : `ws replay`
# les remplace par une attente humaine tirée de la trace (politique « sample », par défaut).
INACTIVE = frozenset({"error", "resume", "idle"})
# Au-delà, une attente humaine est une session laissée ouverte (même seuil que working-set).
PARK_S = 1800.0
# Plafond de la dispersion déduite des percentiles des requêtes chaudes.
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
    hit_rate: float                     # pondéré par les tokens, cache parfait
    cold_fraction: float                # part des requêtes froides
    cold_share_of_prefill: float        # part du prefill non caché portée par les requêtes froides
    warm_uncached: dict[str, int] | None
    mean_gap_s: float | None
    mean_service_s: float | None
    busy_share: float | None            # part du temps où une session a une requête en vol (serveur enregistré)
    resets: int                         # prompts qui ont rétréci (compaction, branche)
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
    # arrondis : garder l'ordre exigé par Percentiles
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
