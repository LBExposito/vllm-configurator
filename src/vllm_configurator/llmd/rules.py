"""Rules of the llm-d EPP target: tune the smart routing between the vLLM instances generated upstream.

Everything is derived from `GenerationResult.facts` (KV tokens per instance, block size, max_num_seqs, number of
instances) and from the workload, never recomputed from scratch (docs/vllm-config-knowledge-base.md §3.5 and §9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .. import formulas
from ..analysis import adjusted_kv_tokens
from ..catalog import Catalog, default_catalog
from ..catalog.schema import Behavior, LlmdVersion
from ..domain import (
    Assumption, Decision, GenerationError, GenerationResult, Indicator, RoutingRequest, RoutingResult, Warning,
)
from ..engine import build_context

KB = "docs/vllm-config-knowledge-base.md"
APPROX_PRODUCER = "approx-prefix-cache-producer"
PRECISE_SCORER = "precise-prefix-cache-scorer"
SCORERS = ["queue-scorer", "kv-cache-utilization-scorer", "prefix-cache-scorer", "no-hit-lru-scorer"]


def _n(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _pct(value: float) -> str:
    return f"{value * 100:.1f} %".replace(".0 %", " %")


def _next_pow2(value: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1, value))))


def _tier(tiers: list[list[float]], x: float) -> int:
    """First weight whose threshold is ≤ x ([[threshold, weight], ...] sorted by decreasing threshold)."""
    for threshold, weight in tiers:
        if x >= threshold:
            return int(weight)
    return int(tiers[-1][1])


@dataclass
class _Out:
    llmd: LlmdVersion
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)
    indicators: list[Indicator] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    comments: dict[str, str] = field(default_factory=dict)
    blocking: list[str] = field(default_factory=list)

    def warn(self, severity: str, message: str, category: str = "routing", source: str | None = None,
             rule_id: str | None = None) -> None:
        self.warnings.append(Warning(severity=severity, category=category, message=message, source=source,
                                     rule_id=rule_id))

    def to_verify(self, item: Behavior, what: str, rule_id: str) -> None:
        if item.status == "unverified":
            self.warn("warning", f"{what} : donnée non confirmée ({item.source}).", category="to_verify",
                      rule_id=rule_id)

    def heuristic(self, name: str) -> Any:
        return self.llmd.heuristics[name].value

    def behavior(self, name: str) -> Any:
        return self.llmd.behaviors[name].value


def resolve_version(routing: RoutingRequest, catalog: Catalog) -> LlmdVersion:
    if not catalog.llmd_versions:
        raise GenerationError(["Aucune version d'llm-d au catalogue (catalog/llmd/)."])
    version = routing.llmd_version or sorted(catalog.llmd_versions)[-1]
    if version not in catalog.llmd_versions:
        raise GenerationError([f"llm-d {version!r} absent du catalogue (catalog/llmd/). Disponibles :"
                               f" {sorted(catalog.llmd_versions)}"])
    return catalog.llmd_versions[version]


def generate_routing(result: GenerationResult, catalog: Catalog | None = None) -> RoutingResult:
    """EPP configuration for the instances of `result`. Raises GenerationError if llm-d is not in the catalog."""
    catalog = catalog or default_catalog()
    request = result.request
    routing = request.routing or RoutingRequest()
    llmd = resolve_version(routing, catalog)
    out = _Out(llmd=llmd)
    precise = routing.prefix_mode == "precise"
    base = dict(llmd_version=llmd.id, prefix_mode=routing.prefix_mode)
    if result.blocking_errors:
        return RoutingResult(**base, decisions=[], warnings=[], indicators=[], assumptions=[],
                             blocking_errors=["La configuration vLLM est incomplète : l'EPP dépend de sa taille de KV"
                                              " et de sa taille de bloc."])

    ctx = build_context(request, catalog)
    facts, wl = result.facts, request.workload
    n = facts.num_instances
    profile = catalog.workload.profiles[wl.usage_type]

    if n == 1:
        out.warn("info", "Une seule instance : l'EPP n'a rien à départager. La configuration reste utile pour le jour où"
                         " les instances se multiplient, et elle est recalculée avec leur nombre.", rule_id="llmd.instances")

    # --- Données d'entrée : hit rate, KV, longueurs --------------------------------------------------------------
    hit = wl.prefix_cache_hit_rate
    if hit is None and profile.prefix_cache_hit_rate is not None:
        hit = profile.prefix_cache_hit_rate.value
        out.assumptions.append(Assumption(
            key="prefix_cache_hit_rate", label="Hit rate attendu avec affinité", value=hit,
            reason=f"Non fourni : valeur publiée pour « {profile.label} » ({profile.prefix_cache_hit_rate.source}).",
            status=profile.prefix_cache_hit_rate.status))
    kv_tokens, kv_note = adjusted_kv_tokens(ctx, {"kv_tokens": facts.kv_tokens_per_instance,
                                                  "gpu_memory_utilization": facts.gpu_memory_utilization})
    isl_mean = formulas.mean_from_percentiles(wl.isl.p50, wl.isl.p95, wl.isl.p99)
    occupancy = _kv_occupancy(result)

    # --- Taille de bloc de l'index ------------------------------------------------------------------------------
    block = _block_size(out, facts, precise)
    if block is None:
        return _finish(out, base, routing, precise, result, None, None)

    window = None
    lru = None
    if not precise:
        lru = _lru_capacity(out, kv_tokens, kv_note, facts, block, n)
        _auto_tune(out)
        window = _match_window(out, wl, routing, block, request.max_model_len or ctx.model.architecture.max_position_embeddings)
    else:
        _precise_vllm_side(out, routing, ctx, request)

    # --- Poids du profil ----------------------------------------------------------------------------------------
    gain = _weights(out, routing, hit, isl_mean, n, occupancy, facts, wl, precise)
    _mode_advice(out, precise, occupancy)
    _overrides(out, routing, precise, kv_tokens, facts)
    by_key = {d.key: d for d in out.decisions}
    block = by_key["tokenProcessorConfig.blockSize" if precise else "blockSizeTokens"].value
    lru = by_key["lruCapacityPerServer"].value if "lruCapacityPerServer" in by_key else None
    window = by_key["maxPrefixTokensToMatch"].value if "maxPrefixTokensToMatch" in by_key else None
    _observed(out, routing, kv_tokens, hit, wl, gain)
    _indicators(out, kv_tokens, block, lru, window, n, hit, gain, occupancy, facts, wl)
    return _finish(out, base, routing, precise, result, lru, window)


def _kv_occupancy(result: GenerationResult) -> float | None:
    plan = result.analysis.test_plan if result.analysis else None
    if plan is None:
        return None
    step = next((s for s in plan.steps if s.is_generated_config), None)
    return None if step is None or step.kv_usage_pct is None else step.kv_usage_pct / 100


def _block_size(out: _Out, facts, precise: bool) -> int | None:
    vllm_block = facts.block_size
    minimum = out.behavior("min_block_size_tokens")
    default = out.llmd.plugins[APPROX_PRODUCER].parameters["blockSizeTokens"].value
    source = {"startup_log": "lue dans les logs de démarrage", "backend_forced": "imposée par le backend d'attention"}
    if precise:
        if vllm_block is None:
            out.blocking.append(
                "Mode precise : tokenProcessorConfig.blockSize doit être égal à la taille de bloc KV de vLLM. Saisissez"
                " « Setting kv cache block size to N » lu dans les logs de démarrage.")
            return None
        out.decisions.append(Decision(
            key="tokenProcessorConfig.blockSize", value=vllm_block, rule_id="llmd.block_size",
            title="Taille de bloc du scorer precise",
            justification=(f"Égale à la taille de bloc KV de vLLM ({vllm_block}, {source.get(facts.block_size_source, 'connue')}) :"
                           " les KV events publient des hashes de blocs vLLM, que le scorer doit recalculer à l'identique."),
            formula="tokenProcessorConfig.blockSize = block_size vLLM", confidence="unverified",
            sources=[f"{KB} §9.1", out.llmd.plugins[PRECISE_SCORER].source]))
        out.comments["tokenProcessorConfig.blockSize"] = "= taille de bloc KV de vLLM"
        return vllm_block

    if vllm_block is not None:
        value = vllm_block if vllm_block >= minimum else math.ceil(minimum / vllm_block) * vllm_block
        aligned = value == vllm_block
        justification = (
            f"Taille de bloc KV de vLLM : {vllm_block} ({source.get(facts.block_size_source, 'connue')}). "
            + ("Un bloc de l'index = un bloc du cache vLLM : la LRU se dimensionne directement en blocs vLLM."
               if aligned else
               f"Elle est sous le minimum de l'EPP ({minimum}), qui l'écraserait : multiple de {vllm_block} juste au-dessus.")
        )
        computation = (f"block_size vLLM = {vllm_block} ≥ {minimum}" if aligned
                       else f"⌈{minimum} / {vllm_block}⌉ × {vllm_block} = {value}")
    else:
        value = minimum
        justification = (f"Taille de bloc vLLM inconnue : minimum de l'EPP ({minimum}). Relevez « Setting kv cache block"
                         " size to N » dans les logs de démarrage pour aligner l'index sur les blocs vLLM.")
        computation = f"minimum EPP = {minimum}"
        out.assumptions.append(Assumption(key="blockSizeTokens", label="Taille de bloc de l'index EPP", value=value,
                                          reason="Taille de bloc vLLM non lue dans les logs."))
    out.decisions.append(Decision(
        key="blockSizeTokens", value=value, rule_id="llmd.block_size", title="Taille de bloc de l'index de préfixes",
        justification=justification + f" Défaut EPP {default}, relevé à {minimum} au démarrage (issue llm-d-router#1158).",
        formula="blockSizeTokens = block_size vLLM, au moins le minimum de l'EPP", computation=computation,
        default=default, sources=[f"{KB} §3.5, §9.3", out.llmd.behaviors["min_block_size_tokens"].source]))
    out.comments["blockSizeTokens"] = "= taille de bloc KV de vLLM" if vllm_block == value else f"minimum EPP {minimum}"
    return value


def _lru_capacity(out: _Out, kv_tokens: int | None, kv_note: str | None, facts, block: int, n: int) -> int | None:
    default = out.llmd.plugins[APPROX_PRODUCER].parameters["lruCapacityPerServer"].value
    if kv_tokens is None:
        out.blocking.append("Taille du KV par instance inconnue : lruCapacityPerServer ne peut pas être calculé."
                            " Saisissez « GPU KV cache size » lu dans les logs de démarrage.")
        return None
    lru = formulas.lru_capacity_per_server(kv_tokens, block)
    estimate = facts.kv_tokens_source != "startup_log" or kv_note is not None
    tracked_default = default * block
    out.decisions.append(Decision(
        key="lruCapacityPerServer", value=lru, rule_id="llmd.lru_capacity", title="Capacité de l'index LRU par pod",
        justification=(
            f"L'index doit suivre tout le KV d'un pod ({_n(kv_tokens)} tokens). Trop petit, l'EPP oublie des préfixes"
            " encore en cache dans vLLM (faux misses : routage au hasard, prefill complet, cache dupliqué) ; trop grand,"
            " il croit à des préfixes déjà évincés (faux hits, moins grave). Le défaut"
            f" ({_n(default)} blocs × {block} = {_n(tracked_default)} tokens) ne couvre que"
            f" {_pct(min(1.0, tracked_default / kv_tokens))} de ce KV." + (f" {kv_note}" if kv_note else "")),
        formula="lruCapacityPerServer = KV_tokens / blockSizeTokens",
        computation=f"{_n(kv_tokens)} / {block} = {_n(lru)}",
        confidence="unverified" if estimate else "verified", default=default,
        sources=[f"{KB} §3.5, §9.2"]))
    out.comments["lruCapacityPerServer"] = f"{_n(kv_tokens)} / {block}"
    if estimate:
        out.warn("warning", "lruCapacityPerServer repose sur une taille de KV estimée : relisez « GPU KV cache size » après"
                            " le déploiement de la config vLLM, l'app recalcule l'index.", category="to_verify",
                 rule_id="llmd.lru_capacity")
    entries = lru * n * out.behavior("lru_bytes_per_entry")
    if entries > 1e9:
        out.warn("warning", f"L'index LRU de l'EPP pèse ~{entries / 1e9:.1f} Go en mémoire ({_n(lru)} entrées × {n} pods) :"
                            " prévoir la mémoire du pod EPP.", rule_id="llmd.lru_capacity")
    return lru


def _auto_tune(out: _Out) -> None:
    param = out.llmd.plugins[APPROX_PRODUCER].parameters["autoTune"]
    out.decisions.append(Decision(
        key="autoTune", value=False, rule_id="llmd.auto_tune", title="autoTune du prefix cache",
        justification=("Désactivé : lruCapacityPerServer est fixé à partir du KV réel. L'autoTune est censé lire"
                       " vllm:cache_config_info (num_gpu_blocks), mais sa correction effective n'est pas confirmée et,"
                       " dans le cas de référence, l'index restait à 2M tokens pour 86M de KV."),
        default=param.value, confidence="unverified", sources=[f"{KB} §9.3, §9.4, §11 point 5",
                                                               out.llmd.behaviors["auto_tune_reads_cache_config_info"].source]))
    out.comments["autoTune"] = "capacité fixée ci-dessous"


def _match_window(out: _Out, wl, routing: RoutingRequest, block: int, max_len: int) -> int:
    default = out.llmd.plugins[APPROX_PRODUCER].parameters["maxPrefixTokensToMatch"].value
    p95 = wl.isl.p95
    if p95 is None:
        p95 = wl.isl.p50
        out.assumptions.append(Assumption(key="isl_p95_window", label="ISL P95 (fenêtre de comparaison)", value=p95,
                                          reason="ISL P95 non fourni : la fenêtre est dimensionnée sur le P50."))
    shared = routing.shared_prefix_tokens or 0
    need = max(p95, shared + block)
    value = max(default, _next_pow2(need))
    value = min(value, max(default, _next_pow2(max_len)))
    computation = f"max({_n(default)}, 2^⌈log2(max(ISL P95 {_n(p95)}, préfixe commun {_n(shared)} + {block}))⌉) = {_n(value)}"
    justification = (
        f"La fenêtre doit dépasser le préfixe commun (system prompt, outils) et atteindre la partie propre à chaque session,"
        f" sinon tous les pods ont le même score. Elle couvre ici l'ISL P95 ({_n(p95)} tokens)"
        + (f" et le préfixe commun ({_n(shared)})" if shared else "") + ". Comparer plus loin ne change pas le choix du pod"
        " pour la queue de distribution et coûte du hachage par requête.")
    out.decisions.append(Decision(
        key="maxPrefixTokensToMatch", value=value, rule_id="llmd.match_window", title="Fenêtre de comparaison des préfixes",
        justification=justification, formula="maxPrefixTokensToMatch > préfixe commun + partie discriminante (≥ ISL P95)",
        computation=computation, default=default, sources=[f"{KB} §3.5, §9.3"]))
    blocks = value // block
    out.decisions.append(Decision(
        key="maxPrefixBlocksToMatch", value=blocks, rule_id="llmd.match_window", title="Fenêtre de comparaison, en blocs",
        justification="Même fenêtre exprimée en blocs : posée explicitement pour que les deux paramètres concordent quelle"
                      " que soit celle que lit la version d'llm-d.",
        formula="maxPrefixBlocksToMatch = maxPrefixTokensToMatch / blockSizeTokens", computation=f"{_n(value)} / {block} = {_n(blocks)}",
        default=out.llmd.plugins[APPROX_PRODUCER].parameters["maxPrefixBlocksToMatch"].value, sources=[f"{KB} §9.3"]))
    out.comments["maxPrefixTokensToMatch"] = f"≥ ISL P95 ({_n(p95)})"
    if shared and shared >= p95:
        out.warn("warning", f"Le préfixe commun ({_n(shared)} tokens) dépasse l'ISL P95 : vérifiez la saisie, la plupart des"
                            " requêtes seraient entièrement communes.", rule_id="llmd.match_window")
    return value


def _precise_vllm_side(out: _Out, routing: RoutingRequest, ctx, request) -> None:
    flag = "--kv-events-config"
    if flag not in ctx.extracted.cli_flags:
        out.blocking.append(f"Mode precise : {flag} absent de vLLM {request.vllm_version}.")
        return
    if not routing.kv_events_endpoint:
        out.blocking.append("Mode precise : indiquez l'endpoint ZMQ des KV events (ex. tcp://<service-epp>:5557).")
        return
    value = {"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": routing.kv_events_endpoint,
             "topic": f"kv@$(POD_IP)@{ctx.model.hf_repo}"}
    out.decisions.append(Decision(
        key="kv_events_config", value=value, rule_id="llmd.precise", title="KV events publiés par vLLM",
        justification=("Mode precise : chaque pod publie ses créations et évictions de blocs en ZMQ, l'EPP tient un index"
                       " exact. $(POD_IP) est expansé par Kubernetes : la variable POD_IP doit être posée sur le conteneur"
                       " (downward API, status.podIP)."),
        cli=flag, cli_kind="json", confidence="unverified", sources=[f"{KB} §9.1", out.llmd.plugins[PRECISE_SCORER].source]))
    out.warn("warning", "Mode precise : clés de --kv-events-config, sidecar tokenizer, indexer et graine de hachage des pods"
                        " (PYTHONHASHSEED identique partout) à confirmer dans le guide llm-d de la version déployée.",
             category="to_verify", rule_id="llmd.precise")


def _weights(out: _Out, routing: RoutingRequest, hit: float | None, isl_mean: float, n: int, occupancy: float | None,
             facts, wl, precise: bool) -> float | None:
    defaults = out.llmd.default_profile.weights
    src = [f"{KB} §9.1, §9.4"]

    # Affinité : gain de prefill apporté par le routage par préfixe.
    gain = None
    shared = routing.shared_prefix_tokens
    if hit is not None:
        if shared is None:
            out.assumptions.append(Assumption(
                key="shared_prefix_tokens", label="Préfixe commun à toutes les sessions", value=0,
                reason="Non fourni : on suppose que tout le préfixe réutilisé est propre à la session (cas où l'affinité"
                       " compte le plus)."))
        shared_hit = min(hit, (shared or 0) / isl_mean)
        hit_random = shared_hit + (hit - shared_hit) / n
        gain = (1 - hit_random) / max(1 - hit, 1e-3)
        weight = _tier(out.heuristic("prefix_weight_by_prefill_gain"), gain)
        justification = (
            f"Sans affinité, une requête n'a qu'une chance sur {n} de retrouver la partie de son préfixe propre à la session :"
            f" le hit rate passerait de {_pct(hit)} à ~{_pct(hit_random)} et le prefill non caché serait multiplié par"
            f" {gain:.1f}. " + ("L'affinité prime." if weight >= 5 else "L'affinité aide sans dominer."
                                if weight >= 3 else "L'affinité rapporte peu : la charge doit primer."))
        computation = (f"h_sans = {shared_hit:.3f} + ({hit:.3f} − {shared_hit:.3f}) / {n} = {hit_random:.3f} ;"
                       f" gain = (1 − {hit_random:.3f}) / (1 − {hit:.3f}) = {gain:.1f}")
        formula = "gain = (1 − h_sans_affinité) / (1 − h), h_sans = h_commun + (h − h_commun) / N"
    else:
        weight = defaults["prefix-cache-scorer"]
        justification = "Hit rate inconnu pour ce type d'usage : poids par défaut de RHOAI. Mesurez le hit rate par pod."
        computation, formula = None, None
    _weight(out, routing, "prefix-cache-scorer", weight, justification, formula, computation, src,
            "Affinité de préfixe" + (" (precise)" if precise else ""))

    # Occupation du KV.
    if occupancy is not None:
        weight = _tier(out.heuristic("kv_weight_by_occupancy"), occupancy)
        justification = (f"Au palier de la config générée, le KV est occupé à ~{_pct(occupancy)}. "
                         + ("Il est presque plein : éviter les préemptions doit compter autant que l'affinité."
                            if weight >= 3 else "Occupation moyenne : ce scorer départage utilement les pods."
                            if weight >= 2 else "Le KV est large : ce scorer ne départage presque rien."))
        computation = f"occupation {_pct(occupancy)} → poids {weight}"
    else:
        weight = defaults["kv-cache-utilization-scorer"]
        justification, computation = "Occupation du KV non estimée : poids par défaut de RHOAI.", None
    _weight(out, routing, "kv-cache-utilization-scorer", weight, justification, "paliers d'occupation du KV", computation,
            src, "Occupation du KV")

    # File d'attente.
    q = out.heuristic("queue_weight")
    if wl.target_concurrency and facts.max_num_seqs:
        load = wl.target_concurrency / n / facts.max_num_seqs
        saturated = load >= out.heuristic("queue_saturation_ratio")
        weight = q["saturated"] if saturated else q["default"]
        justification = (f"{_n(wl.target_concurrency)} requêtes en vol sur {n} pods, soit {_pct(load)} de max-num-seqs"
                         f" ({facts.max_num_seqs}) par pod. "
                         + ("Des files vont se former : l'affinité doit céder plus vite devant un pod chargé (TTFT)."
                            if saturated else "Les pods ne saturent pas : poids par défaut."))
        computation = f"{_n(wl.target_concurrency)} / {n} / {facts.max_num_seqs} = {load:.2f}"
    else:
        weight = q["default"]
        justification, computation = "Requêtes en vol visées non fournies : poids par défaut de RHOAI.", None
    _weight(out, routing, "queue-scorer", weight, justification, "charge par pod / max-num-seqs", computation, src,
            "File d'attente")

    if not precise:
        _weight(out, routing, "no-hit-lru-scorer", out.heuristic("no_hit_lru_weight"),
                "Une requête sans préfixe connu (nouvelle session) part vers le pod le moins récemment choisi : sans lui,"
                " toutes les nouvelles sessions ont le même score et s'empilent.", None, None, src, "Nouvelles sessions")
    else:
        out.warn("info", "Mode precise : no-hit-lru-scorer n'est pas posé, son producer est approx-prefix-cache-producer."
                         " À réintroduire si la version d'llm-d l'accepte avec le scorer precise.", category="to_verify",
                 rule_id="llmd.weights")
    return gain


def _weight(out: _Out, routing: RoutingRequest, scorer: str, computed: int, justification: str, formula, computation,
            sources, title: str) -> None:
    forced = routing.weights.get(scorer)
    value = computed if forced is None else forced
    if forced is not None and forced != computed:
        justification = f"Forcé à {forced} (calculé : {computed}). " + justification
    out.decisions.append(Decision(
        key=f"weight:{scorer}", value=value, rule_id="llmd.weights", title=f"Poids {title} ({scorer})",
        justification=justification, formula=formula, computation=computation, confidence="unverified",
        default=out.llmd.default_profile.weights.get(scorer), sources=sources))


def _mode_advice(out: _Out, precise: bool, occupancy: float | None) -> None:
    limit = out.heuristic("kv_occupancy_precise_advice")
    if occupancy is None:
        return
    if not precise and occupancy >= limit:
        out.warn("warning", f"KV occupé à ~{_pct(occupancy)} : les évictions seront fréquentes et l'index approximatif,"
                            " qui ne les voit pas, va diverger (faux hits). Le mode precise (KV events) devient utile.",
                 rule_id="llmd.mode")
    elif precise and occupancy < limit:
        out.warn("info", f"KV occupé à ~{_pct(occupancy)} : les évictions sont rares et l'index approximatif reste fidèle."
                         " Le mode precise ajoute un sidecar et des KV events pour un gain faible.", rule_id="llmd.mode")


def _observed(out: _Out, routing: RoutingRequest, kv_tokens: int | None, hit: float | None, wl, gain) -> None:
    obs = routing.observed
    if obs is None:
        return
    minimum = out.behavior("min_block_size_tokens")
    if obs.block_size_tokens is not None and obs.block_size_tokens < minimum:
        out.warn("warning", f"Log EPP : blockSizeTokens {obs.block_size_tokens} < minimum {minimum}, l'EPP l'écrase."
                            " La configuration déployée n'est pas celle proposée ici.", rule_id="llmd.observed")
    if obs.lru_capacity_per_server and kv_tokens:
        block = obs.block_size_tokens if obs.block_size_tokens and obs.block_size_tokens >= minimum else minimum
        tracked = obs.lru_capacity_per_server * block
        coverage = tracked / kv_tokens
        out.indicators.append(Indicator(
            key="observed_coverage", label="Couverture du KV par l'index déployé", value=round(coverage * 100, 1), unit="%",
            formula="lruCapacityPerServer × blockSizeTokens / KV_tokens",
            computation=f"{_n(obs.lru_capacity_per_server)} × {block} / {_n(kv_tokens)}",
            interpretation="< 100 % : l'EPP oublie des préfixes encore en cache (faux misses)."))
        if coverage < 0.9:
            out.warn("critical" if coverage < 0.5 else "warning",
                     f"Index déployé : {_n(obs.lru_capacity_per_server)} blocs × {block} = {_n(tracked)} tokens suivis pour"
                     f" {_n(kv_tokens)} de KV : ~{_pct(1 - coverage)} du cache est oublié par l'EPP. Appliquez"
                     " lruCapacityPerServer proposé.", rule_id="llmd.observed")
    if obs.max_prefix_tokens_to_match and wl.isl.p95 and obs.max_prefix_tokens_to_match < wl.isl.p95:
        out.warn("info", f"Fenêtre déployée ({_n(obs.max_prefix_tokens_to_match)} tokens) sous l'ISL P95 ({_n(wl.isl.p95)}).",
                 rule_id="llmd.observed")
    if obs.prefix_cache_hit_rate is not None and hit is not None:
        gap = hit - obs.prefix_cache_hit_rate
        if gap > out.heuristic("hit_rate_gap_alert"):
            out.warn("warning",
                     f"Hit rate mesuré {_pct(obs.prefix_cache_hit_rate)} pour {_pct(hit)} attendu. Causes, dans l'ordre :"
                     " index LRU trop petit (faux misses), affinité trop faible face à la file (poids prefix-cache vs"
                     " queue), évictions dans vLLM (KV plein), redémarrages de pods (index vide).",
                     rule_id="llmd.observed")


def _coerce(name: str, value: Any, default: Any) -> Any:
    """Check a forced parameter against the type of its EPP default. Raises ValueError with a readable message."""
    if isinstance(default, bool) or name == "autoTune":
        if not isinstance(value, bool):
            raise ValueError(f"{name} attend true ou false")
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} attend un entier > 0")
    return value


def _overrides(out: _Out, routing: RoutingRequest, precise: bool, kv_tokens: int | None, facts) -> None:
    """Apply the parameters forced by the user, then check what they break."""
    if not routing.parameters:
        return
    plugin = PRECISE_SCORER if precise else APPROX_PRODUCER
    known = out.llmd.plugins[plugin].parameters
    for name, raw in routing.parameters.items():
        if name not in known:
            out.warn("critical", f"Paramètre llm-d {name!r} inconnu de {plugin} dans le catalogue {out.llmd.id} : ignoré."
                                 f" Connus : {', '.join(known)}.", category="extra_args", rule_id="llmd.overrides")
            continue
        try:
            value = _coerce(name, raw, known[name].value)
        except ValueError as exc:
            out.warn("critical", f"Paramètre llm-d forcé ignoré : {exc} (reçu {raw!r}).", category="extra_args",
                     rule_id="llmd.overrides")
            continue
        index = next((i for i, d in enumerate(out.decisions) if d.key == name), None)
        if index is None:
            out.decisions.append(Decision(
                key=name, value=value, rule_id="llmd.overrides", title=f"{name} (forcé)",
                justification="Forcé par l'utilisateur. " + known[name].description, default=known[name].value,
                confidence="unverified", sources=[known[name].source]))
            out.comments[name] = "forcé"
            continue
        computed = out.decisions[index]
        if computed.value == value:
            continue
        out.decisions[index] = computed.model_copy(update={
            "value": value, "rule_id": "llmd.overrides", "confidence": "unverified",
            "justification": f"Forcé à {value} (calculé : {computed.value}). {computed.justification}"})
        out.comments[name] = f"forcé (calculé : {computed.value})"

    by_key = {d.key: d.value for d in out.decisions}
    minimum = out.behavior("min_block_size_tokens")
    block = by_key.get("blockSizeTokens")
    if not precise and block is not None and block < minimum:
        out.warn("warning", f"blockSizeTokens {block} < minimum {minimum} : l'EPP le relèvera à {minimum} au démarrage,"
                            " et lruCapacityPerServer sera compté en blocs de cette taille.", rule_id="llmd.overrides")
        block = minimum
    if precise and facts.block_size and by_key.get("tokenProcessorConfig.blockSize") != facts.block_size:
        out.warn("critical", f"tokenProcessorConfig.blockSize ≠ taille de bloc vLLM ({facts.block_size}) : les hashes des"
                             " KV events ne correspondront pas, le scorer precise ne verra aucun préfixe.",
                 rule_id="llmd.overrides")
    lru = by_key.get("lruCapacityPerServer")
    if lru and block and kv_tokens and lru * block < kv_tokens:
        coverage = lru * block / kv_tokens
        out.warn("critical" if coverage < 0.5 else "warning",
                 f"lruCapacityPerServer forcé : {_n(lru)} × {block} = {_n(lru * block)} tokens suivis pour {_n(kv_tokens)}"
                 f" de KV, ~{_pct(1 - coverage)} du cache sera oublié par l'EPP (faux misses).", rule_id="llmd.overrides")
    if by_key.get("autoTune") is True and "lruCapacityPerServer" in routing.parameters:
        out.warn("info", "autoTune activé avec une capacité forcée : l'EPP peut remplacer lruCapacityPerServer.",
                 rule_id="llmd.overrides")
    tokens, blocks = by_key.get("maxPrefixTokensToMatch"), by_key.get("maxPrefixBlocksToMatch")
    if tokens and blocks and block and blocks * block != tokens:
        out.warn("warning", f"maxPrefixBlocksToMatch × blockSizeTokens = {_n(blocks * block)} ≠ maxPrefixTokensToMatch"
                            f" ({_n(tokens)}) : la plus petite des deux limites s'applique probablement.",
                 category="to_verify", rule_id="llmd.overrides")
    shared = routing.shared_prefix_tokens
    if tokens and shared and tokens <= shared:
        out.warn("critical", f"maxPrefixTokensToMatch ({_n(tokens)}) ne dépasse pas le préfixe commun ({_n(shared)}) :"
                             " tous les pods auront le même score, l'affinité de session est perdue.",
                 rule_id="llmd.overrides")


def _indicators(out: _Out, kv_tokens, block, lru, window, n, hit, gain, occupancy, facts, wl) -> None:
    add = out.indicators.append
    if lru is not None:
        add(Indicator(key="tracked_tokens", label="Tokens suivis par l'index, par pod", value=lru * block, unit="tokens",
                      formula="lruCapacityPerServer × blockSizeTokens", computation=f"{_n(lru)} × {block}",
                      interpretation=f"Couvre le KV du pod ({_n(kv_tokens)} tokens)."))
        mem = lru * n * out.behavior("lru_bytes_per_entry")
        add(Indicator(key="lru_memory", label="Mémoire de l'index LRU (tous pods)", value=round(mem / 1e6, 1), unit="Mo",
                      formula="lruCapacityPerServer × N × octets par entrée",
                      computation=f"{_n(lru)} × {n} × {out.behavior('lru_bytes_per_entry')} o",
                      interpretation="Ordre de grandeur, à prévoir dans les ressources du pod EPP.",
                      confidence=out.llmd.behaviors["lru_bytes_per_entry"].status))
    if window is not None:
        add(Indicator(key="match_blocks", label="Fenêtre de comparaison", value=window // block, unit="blocs",
                      formula="maxPrefixTokensToMatch / blockSizeTokens", computation=f"{_n(window)} / {block}"))
    if hit is not None and gain is not None:
        add(Indicator(key="prefill_gain", label="Prefill non caché sans affinité / avec", value=round(gain, 1), unit="×",
                      formula="(1 − h_sans_affinité) / (1 − h)",
                      interpretation="Ce que coûterait un routage au hasard : c'est l'enjeu du scorer de préfixe.",
                      confidence="unverified"))
    if occupancy is not None:
        add(Indicator(key="kv_occupancy", label="Occupation du KV au palier de la config", value=round(occupancy * 100, 1),
                      unit="%", interpretation="Large : évictions rares, index approximatif fidèle.", confidence="unverified"))
    if wl.target_concurrency and facts.max_num_seqs:
        per_pod = wl.target_concurrency / n
        add(Indicator(key="load_per_pod", label="Requêtes en vol par pod", value=round(per_pod),
                      formula="requêtes en vol visées / N", computation=f"{_n(wl.target_concurrency)} / {n}",
                      interpretation=f"À comparer à max-num-seqs ({facts.max_num_seqs}) : au-delà, files d'attente."))


def _finish(out: _Out, base: dict, routing: RoutingRequest, precise: bool, result: GenerationResult,
            lru: int | None, window: int | None) -> RoutingResult:
    llmd = out.llmd
    by_key = {d.key: d for d in out.decisions}
    weights = {s: by_key[f"weight:{s}"].value for s in SCORERS if f"weight:{s}" in by_key}
    plugins: list[dict[str, Any]] = []
    if not out.blocking:
        plugin = PRECISE_SCORER if precise else APPROX_PRODUCER
        params: dict[str, Any] = {}
        for name in llmd.plugins[plugin].parameters:
            if name in by_key:
                *path, leaf = name.split(".")
                node = params
                for part in path:
                    node = node.setdefault(part, {})
                node[leaf] = by_key[name].value
        plugins.append({"type": plugin, "parameters": params})
        if precise:
            weights = {(PRECISE_SCORER if k == "prefix-cache-scorer" else k): v for k, v in weights.items()}
        plugins += [{"type": scorer} for scorer in weights if scorer != PRECISE_SCORER]
        plugins.append({"type": llmd.default_profile.picker})
        out.to_verify(llmd.placement, "Emplacement de la config dans le LLMInferenceService", "llmd.placement")
        if not precise:
            out.to_verify(llmd.plugins[APPROX_PRODUCER], f"Plugin {APPROX_PRODUCER} porteur des paramètres", "llmd.producer")
        out.warn("info", "Recalculée à chaque changement de la config vLLM : gpu-memory-utilization, max-num-seqs ou la"
                         " capture des CUDA graphs changent la taille du KV, donc lruCapacityPerServer.", rule_id="llmd.recalc")
    vllm_args = {}
    if "kv_events_config" in by_key:
        vllm_args["--kv-events-config"] = by_key["kv_events_config"].value
    expected = [] if out.blocking else _expected_logs(by_key, weights, precise, llmd)
    severity = {"critical": 0, "warning": 1, "info": 2}
    return RoutingResult(
        **base, decisions=out.decisions, warnings=sorted(out.warnings, key=lambda w: severity[w.severity]),
        indicators=out.indicators, assumptions=out.assumptions, plugins=plugins, weights=weights,
        picker=llmd.default_profile.picker, comments=out.comments, vllm_args=vllm_args, expected_logs=expected,
        blocking_errors=out.blocking)


def _expected_logs(by_key: dict[str, Decision], weights: dict[str, int], precise: bool, llmd: LlmdVersion) -> list[str]:
    short = ", ".join(f"{scorer.removesuffix('-scorer')} {w}" for scorer, w in weights.items())
    lines = []
    if not precise:
        v = {k: by_key[k].value for k in ("autoTune", "blockSizeTokens", "maxPrefixBlocksToMatch",
                                          "maxPrefixTokensToMatch", "lruCapacityPerServer")}
        lines.append(f"Prefix DataProducer initialized config={{autoTune:{str(v['autoTune']).lower()},"
                     f" blockSizeTokens:{v['blockSizeTokens']}, maxPrefixBlocksToMatch:{v['maxPrefixBlocksToMatch']},"
                     f" maxPrefixTokensToMatch:{v['maxPrefixTokensToMatch']}, lruCapacityPerServer:{v['lruCapacityPerServer']}}}")
        lines.append("Absent : « configured blockSizeTokens is below the recommended minimum, overriding it »")
    lines.append(f"EPP config after phase two: {llmd.default_profile.handler} ; Scorers: {short} ;"
                 f" Picker: {llmd.default_profile.picker}")
    return lines
