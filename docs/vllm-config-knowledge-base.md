# Base de connaissances : dimensionnement et configuration vLLM (+ llm-d)

> **But de ce document** : servir de base à Claude Code pour construire une app qui génère une configuration vLLM à partir du profil de charge (ISL/OSL), du matériel (GPU), du modèle et des contraintes de latence. La **v1 ne couvre que vLLM**. La configuration llm-d / EPP (section 9) est prévue pour une v2 et sert ici de contexte.
>
> Ce document résume une session de travail réelle (novembre 2026). Chaque affirmation est classée :
> - ✅ **vérifié** : confirmé par une source (liste en fin de document) ou par les logs réels de l'utilisateur ;
> - 🔶 **à vérifier** : déduit ou non confirmé pour la version utilisée.

---

## 1. Environnement de référence (cas réel)

| Élément | Valeur |
|---|---|
| Plateforme | Red Hat OpenShift AI (RHOAI) **3.5.2**, KServe (`LLMInferenceService`), llm-d (EPP) |
| Moteur | vLLM **0.33.0** (CUDA 13 uniquement) |
| Modèle | DeepSeek V4.1 (build « 1017 »), MoE |
| Topologie | **2 pods indépendants**, chacun en **TP4 / DP1 sur 4× B300**. Pas de DP entre les pods : le routage est fait par l'EPP llm-d. |
| KV cache (log de démarrage) ✅ | `GPU KV cache size: 86,225,834 tokens` ; `Maximum concurrency for 1,048,576 tokens per request: 82.23x` (par pod) |
| Taille de bloc KV (log) ✅ | `Setting kv cache block size to 128 for FLASHMLA_MEGA_ATTN_DSV41/DEEPSEEK_SPARSE_SWA/DEEPSEEK_V41_INDEXER/CompressorBackend` : imposée par le backend, ne pas la forcer |
| Cas d'usage cible | Chat agentique généraliste de type « Claude Cowork » : interactif, appels d'outils, sessions multi-tours |
| Profil mesuré en test | ISL P50 **24K**, P95 **75K**, P99 **666K** ; OSL P50 **4K** (🔶 OSL suspect, voir §6) |

### Configuration initiale de l'utilisateur (avant la session)

```yaml
--max-num-batched-tokens 16384
--max-num-seqs 128
--speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"fly","fly_window_size": 2,"fly_entropy_threshold": 0.3,"fly_entropy_top_k": 3}'
```

---

## 2. Concepts clés

### 2.1 ISL / OSL
- **ISL** (*Input Sequence Length*) : longueur du prompt en tokens (system prompt, définitions d'outils, historique, documents, résultats d'outils).
- **OSL** (*Output Sequence Length*) : nombre de tokens générés, raisonnement compris.
- **ISL effectif** : partie du prompt **absente du prefix cache**. C'est la seule partie recalculée et la seule qui consomme le budget de prefill.
- L'ISL correspond à la phase de **prefill** (traitement massif en parallèle), l'OSL à la phase de **decode** (un token par step et par séquence, ou plus avec le speculative decoding).

### 2.2 `max_num_batched_tokens` (budget de tokens par step) ✅
- **Plafond** de tokens traités par forward pass du scheduler. Ce n'est pas une taille de batch fixe.
- vLLM V1 (chunked prefill par défaut) remplit chaque step dans cet ordre :
  1. d'abord le **decode** : 1 token par séquence en cours (voir §2.4 pour le speculative decoding) ;
  2. ensuite le **prefill** (morceaux des prompts en attente ou partiellement traités) dans le budget restant.
- En decode pur, le budget n'est **jamais** rempli : 128 séquences ≈ 128 tokens par step.
- **Contreparties d'un budget élevé** :
  - un step chargé en prefill est long, donc **pics d'ITL** pour toutes les séquences en decode ;
  - la mémoire d'activations est réservée au démarrage, donc **un peu moins de KV cache**.
- **Intérêt pour un MoE** : avec 128 tokens par step et un routage top-k sur des centaines d'experts, chaque expert ne voit que quelques tokens. Les GEMM sont alors memory-bound. Ce sont les steps chargés en prefill (plusieurs milliers de tokens) qui rendent les experts efficaces en compute.

### 2.3 `max_num_seqs`
- Nombre maximal de séquences actives en même temps dans **une instance** (un engine core).
- Avec DP-attention, la limite s'applique **par rang DP**. Dans le cas de référence (TP4/DP1), elle s'applique au pod entier.
- La valeur par défaut de 128 plafonne le batch. La limite réelle vient soit du KV cache (`KV_tokens / contexte_typique`), soit de la cible d'ITL.

### 2.4 Speculative decoding : DSpark et FLy (vLLM 0.33)
- **DSpark** : drafter à diffusion par blocs (semi-autorégressif), avec une tête de confiance. ✅
  - **Depuis 0.33, DSpark et DFlash ne réservent plus de slots de draft dans le budget de tokens** (PR #59468). Le decode compte donc pour **~1 token par séquence** dans `max_num_batched_tokens`, et non `1+k`. Gain annoncé sur DeepSeek-V4.1-Flash en GB300 : 1 373 → 3 390 tokens/s en sortie. ✅
  - Vérification adaptative (*confidence-scheduled verification*) : le nombre de tokens vérifiés varie selon la requête, et les draft tokens comptent dans le coût du step. 🔶 Le flag d'activation et sa valeur par défaut en 0.33 restent à vérifier.
  - Nouveautés 0.33 : raffineur **XPress** pour les drafts DSpark/DFlash (#54448), option **`draft_lm_head_quantization`** (#59973, environ +18 % sur le decode à concurrence 1). ✅
  - Recommandation du blog vLLM : **`max_cudagraph_capture_size = (k+1) × max_num_seqs`**, et `--gpu-memory-utilization 0.8` si la capture fait déborder la mémoire. ✅
- **FLy** : politique de vérification **lossy**. Elle diffère un rejet à une position à forte entropie si les tokens de draft suivants restent alignés. ✅
  - **Désactivée par défaut** en 0.33 (opt-in, #53987). Gain annoncé : ×1,04 à ×1,31 sur le decode (MI355X, B300).
  - Qualité : au pire ~97,9 % de qualité conservée ; en MRV2, ~96,3 % d'accuracy sur NVIDIA (une bonne réponse de moins sur AIME25).
  - Contraintes : `fly_window_size < num_speculative_tokens`, avec k ≥ 2.
  - **À éviter par défaut pour les usages agentiques** (JSON des tool calls, arguments exacts). Possible en A/B avec mesure du taux de tool calls valides.
- **Longueur d'acceptation moyenne** `a` (tokens émis par step et par séquence) : se mesure via Prometheus, `a = 1 + spec_decode_num_accepted_tokens / spec_decode_num_drafts`.

### 2.5 `--long-prefill-token-threshold` : comportement depuis vLLM 0.31 ✅
- Rôle : plafonner la part du budget qu'un seul prefill peut prendre dans un step, pour qu'un long prompt n'affame pas les autres requêtes.
- **PR #57951** : si une seule requête est présente (running + waiting + skipped_waiting ≤ 1), le plafond est **désactivé** et elle prend tout le budget. Avant, elle était découpée sans raison.
- **PR #58459** : avec plusieurs requêtes, le plafond devient **dynamique**. Exemple de la PR : budget 8192, seuil 2048, deux requêtes ; avant 2048 chacune (batch à moitié vide), maintenant 4096 chacune (batch plein). Formule déduite 🔶 : environ budget ÷ nombre de requêtes éligibles, le seuil servant de garde-fou.
- **Conséquence** : un seuil bas ne gaspille plus de budget, on peut le choisir **uniquement pour l'équité**. Heuristique : **seuil ≈ budget / 4**, soit jusqu'à ~4 prefills en parallèle en cas de contention.
- 🔶 À vérifier dans `vllm/v1/core/sched/scheduler.py` : les séquences en decode entrent-elles dans le diviseur ? Si oui, la part dynamique est minuscule sous forte charge et seul le seuil fixe compte.
- Mécanisme distinct, à ne pas confondre : limites de prefills partiels concurrents (#57427, #49075).

### 2.6 Prefix caching
- Activé par défaut en V1, **indispensable** pour l'agentique (taux de hit de 95 à 98 % mesurés en production, voir §6).
- Seuls des **blocs complets** sont réutilisés (128 tokens ici). L'effet est négligeable sur des prompts de 24K et plus.
- Pour en profiter : system prompt et définitions d'outils **identiques à l'octet près** et placés **en tête** du prompt.
- 0.33 : PCP (prefill context parallel) combinable avec le prefix caching sur DeepSeek-V4.1 (#59857), utile pour le TTFT des très longs prompts. ✅

---

## 3. Formules (cœur de l'app)

Notations : `B` = `max_num_batched_tokens`, `S` = nombre de séquences actives (≤ `max_num_seqs`), `k` = `num_speculative_tokens`, `a` = longueur d'acceptation moyenne, `d` = tokens de decode comptés dans le budget par séquence.

### 3.1 Condition pour remplir le budget de tokens
En régime stable, environ `S·a / OSL` requêtes se terminent à chaque step et sont remplacées par autant de nouvelles, qui apportent chacune `ISL_eff` tokens de prefill :

```
ISL_eff / OSL  ≥  (B − S·d) / (S·a)
```

- Sans speculative decoding : `d = 1`, `a = 1`.
- Speculative decoding qui réserve des slots (MTP/EAGLE, cas général) : `d = 1 + k`.
- DSpark/DFlash en vLLM ≥ 0.33 : `d ≈ 1` (pas de réservation de slots).

| Scénario (B = 16384, S = 128) | Ratio ISL_eff/OSL nécessaire |
|---|---|
| Sans spec decoding | (16384 − 128) / 128 ≈ **127** |
| MTP k=1, a ≈ 1,8 | (16384 − 256) / 230 ≈ **70** |
| DSpark k=3 (0.33), a = 2,5 / 3,0 / 3,5 | ≈ **51 / 42 / 36** |

Prérequis : une **concurrence soutenue ≥ S**, d'après la loi de Little.

### 3.2 Capacité de concurrence du KV cache
```
concurrence_max ≈ KV_tokens / longueur_de_contexte_typique
```
Exemple de référence : 86 225 834 / 24 000 ≈ **3 600** contextes au P50 ; / 75 000 ≈ **1 150** au P95 (par pod).

### 3.3 Capture des CUDA graphs
```
max_cudagraph_capture_size = (k + 1) × max_num_seqs
```
k=3 : 128 → 512, 256 → 1024, 384 → 1536, 512 → 2048.

### 3.4 Coût d'un step de decode
```
tokens_vérifiés_par_step ≈ S × (k + 1)
```
À 512 séquences et k=3, environ 2 000 tokens vérifiés par step de decode, auxquels s'ajoute le prefill. C'est à surveiller pour l'ITL.

### 3.5 Index LRU de l'EPP llm-d (v2)
```
lruCapacityPerServer = KV_tokens / blockSizeTokens
maxPrefixTokensToMatch  >  préfixe_commun (system + tools) + partie discriminante de la session
```
Exemple : 86 225 834 / 128 ≈ **673 000**.

### 3.6 Charge par expert d'un MoE (ordre de grandeur, générique)
```
tokens_par_expert ≈ T × top_k / n_experts
```
Où `T` est le nombre de tokens du step. Les valeurs de `top_k` et `n_experts` dépendent du modèle : 🔶 les spécificités de DeepSeek V4.1 n'ont pas été vérifiées.

---

## 4. Profils de charge et recommandations

| Profil | ISL / OSL typiques | Ratio effectif | Remplit 16K ? | Budget conseillé |
|---|---|---|---|---|
| Chat classique | ~1–6K / ~100–400 | ~4–15 | Non | 4–8K |
| **Agentique multi-tours (type Cowork)** | 24–68K par appel / ~250 par appel, avec 95–98 % de cache hit | **~12–14** (TraceLab) | Non (sauf rafales) | **8K** |
| RAG / Q&A sur documents | 16K / ~150–400 | ~40–100 | Presque | 8–16K |
| Extraction / classification sur longs documents | 32K / ~200–800 | ~40–160 | Oui | 16K+ |
| Résumé de très longs contextes | 128K / ~1K | ~128 | Oui | 16K+ |
| Scoring, rerank, LLM-as-judge, `prompt_logprobs` | variable / 1–20 | très élevé | Oui | 16K+ |
| Batch offline en rafale | — | — | Oui pendant le remplissage | 16K+ |

**Règle générale** : un budget élevé ne paie que s'il y a beaucoup de **prefill non caché**, c'est-à-dire de gros contextes uniques, une sortie courte et une concurrence soutenue. En interactif, le budget arbitre entre **TTFT** (budget haut) et **ITL** (budget bas).

---

## 5. Règles de décision (heuristiques pour l'app v1)

| Paramètre | Règle |
|---|---|
| `max_num_batched_tokens` | Interactif : **8192**. Batch, long contexte ou sortie courte : **16384+**. Vérifier avec le ratio de §3.1. |
| `max_num_seqs` | `min(concurrence_KV au contexte P50, limite ITL)`. Point de départ : 256–512 si le KV est large. |
| `--long-prefill-token-threshold` | ≈ `max_num_batched_tokens / 4` (vLLM ≥ 0.31). |
| `max_cudagraph_capture_size` | `(k+1) × max_num_seqs` (via `--compilation-config`). |
| `--gpu-memory-utilization` | 0,8 avec de grosses captures de CUDA graphs avec spec decoding. À augmenter si aucun OOM et si le KV est limitant. |
| `--kv-cache-dtype` | `fp8` (requis par la disposition `fp8_ds_mla` des modèles DeepSeek). |
| `--block-size` | **Ne pas forcer** si le backend impose une taille (DeepSeek V4.1 : 128). |
| Spec decoding | Méthode native du modèle (DSpark pour DeepSeek V4.x), k = 3. **FLy désactivé** pour l'agentique ou les tool calls. |
| Offload KV vers le CPU | Seulement si `concurrence_KV au contexte typique < sessions actives + sessions en pause`. Inutile dans le cas de référence. |
| P/D disaggregation | Seulement si les gros prefills dégradent l'ITL malgré le budget et le seuil. Sinon, des pods symétriques avec affinité sont plus simples. |
| DP-attention | Utile si le KV est limitant (évite la réplication du KV MLA entre rangs TP). Inutile dans le cas de référence. |

---

## 6. Données de charge réelles (références publiques)

Ni Anthropic ni OpenAI ne publient de distributions de tokens par requête. Meilleures références disponibles : ✅

**GitHub Copilot coding agent (Microsoft Research, juin 2026)**
- Par appel : prompt médian **68K** (dont 63K en cache), sortie médiane **247**, 88 % des appels < 1 000 tokens.
- Par tour : prompt médian 227,6K, P90 1,5M (somme sur plusieurs appels) ; sortie médiane 1,9K.
- Cache hit par appel ~98 % (médiane) ; ~55 % au passage d'un tour à l'autre ; ~8 % après un changement de modèle.
- Appels LLM par session : médiane 15, moyenne 40,6.
- À titre de comparaison, chat : prompt médian ~750, sortie ~105.

**TraceLab (Université de Washington, sessions Claude Code et Codex)**
- 4 300 sessions, ~55 Md de tokens : 52,56 Md en cache, **2,34 Md nouveaux**, **186,9 M générés**.
- Ratio entrée/sortie brut ~294× ; **ratio effectif (nouveaux tokens / générés) ≈ 12,5**.
- Hit rate global 95,7 % (97,5 % pour les continuations d'outils, 84,4 % pour les nouveaux messages utilisateur).
- 88 % des rounds répondent à un résultat d'outil ; préfixe typique de 32 à 256K, nouvelle entrée de quelques centaines à quelques milliers de tokens, sortie de quelques centaines de tokens.
- TTFT par étape ≈ **25 %** du temps de génération d'un round (Codex). Pause utilisateur : médiane ~1,4 min, moyenne 46,7 min.

**OpenRouter, State of AI (100T tokens)**
- Prompt moyen ~1,5K → **6K+** ; sortie moyenne ~150 → **~400**. Les prompts de programmation dépassent régulièrement 20K.

**Diagnostic du profil de test de l'utilisateur**
- ISL P50 24K / P95 75K : **cohérent** pour un usage agentique mixte.
- P99 666K : 🔶 à vérifier. Est-ce une mesure par appel ou un cumul par tour/session ? Sans compactage du contexte, c'est possible avec un contexte de 1M.
- **OSL P50 4K : environ 15× les valeurs réelles par appel.** Causes probables : benchmark synthétique à sortie fixe (`--random-output-len`, `ignore_eos`), mesure par tour, ou thinking long.
- **Recommandation de bench** : rejouer des sessions multi-tours (le contexte s'accumule, +0,5–3K tokens par appel, OSL médian ~250 avec une longue traîne) plutôt que des requêtes indépendantes.

---

## 7. Configuration finale retenue (cas de référence, par pod)

```yaml
args:
  # --- Parallélisme : TP4, DP1 (inchangé) ---

  # --- Scheduler ---
  - --max-num-seqs
  - "512"
  - --max-num-batched-tokens
  - "8192"
  - --long-prefill-token-threshold
  - "2048"

  # --- KV cache / prefix caching ---
  - --enable-prefix-caching
  - --kv-cache-dtype
  - fp8
  - --gpu-memory-utilization
  - "0.8"
  # pas de --block-size : imposé à 128 par le backend DeepSeek V4.1
  # pas d'offload KV vers le CPU : 86M tokens de KV par pod suffisent
  # pas de --kv-events-config : l'EPP RHOAI utilise le scorer approximatif (voir §9)

  # --- Speculative decoding (FLy retiré) ---
  - --speculative-config
  - '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'

  # --- CUDA graphs ---
  - --compilation-config
  - '{"max_cudagraph_capture_size":2048}'   # (k+1) × 512
```

**Changements par rapport à la configuration initiale**

| Paramètre | Avant | Après | Raison |
|---|---|---|---|
| `max-num-batched-tokens` | 16384 | 8192 | ITL plus stable ; le ratio agentique (~12–14) ne remplit pas 16K |
| `max-num-seqs` | 128 | 512 | KV énorme (~3 600 contextes au P50) ; trafic surtout en decode |
| `long-prefill-token-threshold` | — | 2048 | Équité entre une grosse reprise et les petits tours (≥ 0.31) |
| FLy | activé | retiré | Lossy, risqué pour les tool calls |
| Capture des CUDA graphs | — | 2048 | (k+1) × max_num_seqs |

**À surveiller après le passage à 512** : ITL p95. Si elle dépasse la cible, redescendre à 384 séquences et à une capture de 1536. Vérifier aussi qu'il n'y a pas d'OOM pendant la capture.

---

## 8. Notes de version vLLM (impactant la config)

**0.31.0**
- Nouveau comportement de `--long-prefill-token-threshold` (#57951, #58459), voir §2.5.

**0.33.0** ✅
- DSpark/DFlash : plus de réservation de slots de draft (#59468) ; raffineur XPress (#54448) ; `draft_lm_head_quantization` (#59973).
- FLy : opt-in, désactivé par défaut (#53987).
- Lookahead block hashes activés par défaut pour les hits de prefix cache EAGLE/MTP (#61589).
- PCP + prefix caching combinables sur DeepSeek-V4.1 (#59857).
- FA4 sparse MLA decode activé par défaut sur Blackwell (#55866).
- Offload CPU : jusqu'à 32 fichiers shm (#59546).
- Speculators Medusa et MLP supprimés (#61701).
- **Changements cassants** :
  - wheels et images en **CUDA 13.0 uniquement** (#62575) ;
  - **NIXL ≥ 1.5** (#61032) ;
  - le **frontend Rust renvoie une erreur sur les flags `vllm serve` non supportés** au lieu de les ignorer (#62431) ;
  - **`tools: []` vide → HTTP 400** (#63060) ;
  - les chat templates envoyés côté requête exigent `--trust-request-chat-template` (#62673) ;
  - HTTP 503 quand la génération est en pause (#51488) ;
  - `--mamba-cache-mode` déprécié ; LM Format Enforcer déprécié (opt-in en 0.34, supprimé en 0.35).

**Validation des flags (à intégrer dans l'app)**
```bash
vllm serve --help=all | grep -iE "long-prefill|cudagraph|offload|max-num"
```
🔶 Les noms exacts en 0.33 de `--kv-offloading-backend`/`--kv-offloading-size` et de `max_cudagraph_capture_size` n'ont pas été confirmés dans les notes de version.

---

## 9. llm-d / EPP (prévu pour la v2, contexte)

### 9.1 Scorer approximatif vs precise
- **RHOAI 3.5, profil par défaut** ✅ : `queue-scorer` 2, `kv-cache-utilization-scorer` 2, **`prefix-cache-scorer` 3 (approximatif, basé sur des hashes)**, `no-hit-lru-scorer` 2, picker `max-score-picker`.
- Mode approximatif : l'EPP estime le contenu du cache de chaque pod à partir des requêtes qu'il a lui-même routées. **Aucun `--kv-events-config` côté vLLM.**
- Mode precise (`precise-prefix-cache-scorer`) : les pods publient des KV events en ZMQ. Il faut un sidecar tokenizer UDS et un accès internet au démarrage de l'EPP, ainsi que `--block-size` vLLM = `tokenProcessorConfig.blockSize` du scorer. Peu utile quand le KV est énorme, car les évictions sont rares et l'estimation approximative reste fidèle.

### 9.2 Comment fonctionne `lruCapacityPerServer`
- L'EPP découpe chaque prompt routé en blocs de `blockSizeTokens` tokens, calcule un hash chaîné par bloc et les range dans une **LRU propre à chaque pod**.
- À la requête suivante, il compte les blocs consécutifs en correspondance pour chaque pod, ce qui donne le score de préfixe.
- `lruCapacityPerServer` = taille maximale de cette LRU, en blocs. Tokens suivis = capacité × `blockSizeTokens`.
- **Trop petit** : faux misses. L'EPP oublie des préfixes encore présents dans vLLM, le routage revient au hasard, ce qui provoque un prefill complet et un doublon de cache.
- **Trop grand** : faux hits. Moins grave, l'affinité est préservée.
- Coût mémoire : quelques dizaines d'octets par entrée, soit quelques dizaines de Mo pour ~673 000 entrées × 2 pods.

### 9.3 Logs réels de l'EPP (cas de référence) ✅
```
Prefix DataProducer initialized config={autoTune:true, blockSizeTokens:16, blockSize:0,
  maxPrefixBlocksToMatch:2048, maxPrefixTokensToMatch:131072, lruCapacityPerServer:31250}
WARNING: configured blockSizeTokens is below the recommended minimum, overriding it. (16 → 64, issue llm-d-router#1158)
auto-created default producer: approx-prefix-cache-producer (consumer: no-hit-lru-scorer)
auto-created default producer: token-producer (consumer: approx-prefix-cache-producer)
EPP config after phase two: single-profile-handler ; Filters: [] ;
  Scorers: queue 2, kv-cache-utilization 2, prefix-cache 3, no-hit-lru 2 ; Picker: max-score-picker ;
  FlowControlConfig: nil
```
Lecture :
- Fenêtre de comparaison : 128K tokens, suffisante (> P95 de 75K, au-delà du system prompt commun).
- Le scorer travaille **en tokens** (via `token-producer`), et non en caractères comme dans les anciennes versions de GIE.
- **Problème** : `lruCapacityPerServer: 31250` × 64 ≈ **2M tokens suivis** contre **86M réels**. Sans correction par l'autoTune, ~97 % du cache est « oublié ».
- Le flow control est désactivé : acceptable avec 2 pods.

### 9.4 Configuration EPP proposée
```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: approx-prefix-cache-producer   # 🔶 ou prefix-cache-scorer selon la version
    parameters:
      autoTune: false
      blockSizeTokens: 128               # = taille de bloc KV de vLLM
      maxPrefixTokensToMatch: 131072
      lruCapacityPerServer: 673000       # 86 225 834 / 128
  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: prefix-cache-scorer
  - type: no-hit-lru-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
        weight: 2
      - pluginRef: kv-cache-utilization-scorer
        weight: 1        # KV énorme : ce scorer ne départage presque rien
      - pluginRef: prefix-cache-scorer
        weight: 5        # l'affinité prime en agentique
      - pluginRef: no-hit-lru-scorer
        weight: 2
      - pluginRef: max-score-picker
```
- Dans RHOAI, cette configuration va dans la config inline du scheduler du `LLMInferenceService` (doc RHOAI 3.5, section 10.5).
- Vérification après déploiement : plus de warning d'override, `lruCapacityPerServer: 673000`, poids 2 / 1 / 5 / 2 dans « EPP config after phase two ».
- **À recalculer** si `gpu-memory-utilization`, `max-num-seqs` ou la capture des CUDA graphs changent, car la taille du KV change avec.
- Pourquoi l'affinité est critique : deux pods distincts ne partagent aucun cache. Sans affinité, le hit rate peut passer de ~95 % à ~50 % et le prefill réel être multiplié par 10.

---

## 10. Métriques à surveiller

| Métrique | Usage |
|---|---|
| `vllm:iteration_tokens_total` (histogramme) | Distribution réelle des tokens par step : le budget est-il rempli ? |
| Débit de prompt tokens (non cachés) / débit de generation tokens | Ratio effectif réel, à comparer à §3.1 |
| `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` (par pod) | Cible > 90 % en agentique, sinon problème de routage ou d'éviction |
| TTFT p95 / ITL p95 | Arbitrage du budget de tokens et de `max_num_seqs` |
| `num_requests_running` / `num_requests_waiting`, préemptions | Saturation du KV ou du compute |
| `spec_decode_num_accepted_tokens` / `spec_decode_num_drafts` | Longueur d'acceptation `a` |
| `vllm:cache_config_info` (labels `block_size`, `num_gpu_blocks`) | Données lues par l'autoTune de l'EPP |

**Test de validation du seuil de prefill** : ~100 sessions en decode en fond, puis 1 prompt non caché de 100K, puis immédiatement 5 à 10 prompts de 2K. Comparer le TTFT des petits prompts avec un seuil de 2048 puis de 4096.

---

## 11. Points ouverts / à vérifier 🔶

1. Noms exacts des flags en 0.33 : offload KV, `max_cudagraph_capture_size`. Le frontend Rust échoue sur les flags inconnus.
2. Les séquences en decode entrent-elles dans le diviseur du seuil dynamique (#58459) ?
3. Activation et valeur par défaut de la vérification adaptative DSpark en 0.33.
4. Emplacement exact des paramètres du producer dans la version d'llm-d livrée avec RHOAI 3.5.2 (`approx-prefix-cache-producer` ou `prefix-cache-scorer`).
5. L'autoTune de l'EPP corrige-t-il réellement `lruCapacityPerServer` depuis `vllm:cache_config_info` ?
6. Spécificités d'architecture de DeepSeek V4.1 (nombre d'experts, top-k, taille du KV par token).
7. Nature de l'OSL P50 de 4K dans les tests (sortie fixe ou réelle) et du P99 de 666K (par appel ou cumulé).

---

## 12. Proposition de cahier des charges pour l'app (v1 : vLLM seul)

> Suggestion de départ, que Claude Code peut adapter.

### 12.1 Entrées
- **Matériel** : modèle de GPU (HBM, génération : Hopper/Blackwell/B300…), nombre de GPU par instance, nombre d'instances.
- **Modèle** : poids (taille, quantization), architecture (dense/MoE, type d'attention : MHA/GQA/MLA/sparse), KV bytes par token, contraintes de taille de bloc imposées par le backend, méthode de spec decoding native (MTP/EAGLE/DSpark/DFlash), contexte maximal.
- **Version vLLM** : elle conditionne le comportement (§8). Exemple : réservation des slots de draft selon la méthode et la version, sémantique du seuil de long prefill.
- **Profil de charge** : ISL et OSL en percentiles (P50/P95/P99), taux de prefix cache hit attendu (ou type d'usage : chat / agentique / RAG / batch), concurrence cible, mode interactif ou batch.
- **SLO** : TTFT p95 et ITL p95 cibles.
- **Tolérance lossy** : FLy autorisé ou non.
- **Optionnel** : valeur réelle `GPU KV cache size` lue dans les logs, qui remplace l'estimation.

### 12.2 Sorties
- Arguments `vllm serve` (format YAML `args` pour KServe/K8s et format CLI).
- Justification de chaque valeur (formule et règle appliquées, §3 et §5).
- Avertissements : FLy lossy, OSL suspect, KV limitant, flags à vérifier selon la version, changements cassants (§8).
- Indicateurs calculés : ratio ISL_eff/OSL réel contre ratio nécessaire, concurrence permise par le KV, taille de capture des CUDA graphs, tokens par step de decode.
- Commande de vérification des flags et liste des métriques à surveiller.

### 12.3 Architecture suggérée
- **Catalogues de données versionnés** (JSON/YAML), séparés du moteur de règles : GPU, modèles, versions de vLLM (flags disponibles, défauts, changements de comportement).
- **Moteur de règles pur** (fonctions testables) qui implémente §3 et §5.
- **Tests de non-régression** : le cas de référence (§1 et §7) sert de « golden test ». Avec les entrées de §1, l'app doit produire la configuration de §7, ou une configuration justifiée qui s'en écarte.
- Extensibilité prévue pour la v2 : module llm-d/EPP (§9) qui réutilise `KV_tokens` et la taille de bloc calculés en v1.

---

## Sources

**vLLM**
- [FLy Verification – vLLM docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/fly/)
- [Adaptive Verification in vLLM: DSpark confidence-scheduled verification (blog vLLM)](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification)
- [Adaptive Verification Without a Confidence Head (blog vLLM)](https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator)
- [DSpark paper (arXiv 2607.05147)](https://arxiv.org/pdf/2607.05147)
- [vLLM API – deepseek_v41 dspark](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/amd/dspark/)
- [vLLM PR #57951 – long_prefill_token_threshold, requête seule](https://github.com/vllm-project/vllm/pull/57951)
- [vLLM PR #58459 – long_prefill_token_threshold dynamique](https://github.com/vllm-project/vllm/pull/58459)
- [vLLM releases – GitHub](https://github.com/vllm-project/vllm/releases)
- [vLLM v0.33.0 – ai-tldr](https://ai-tldr.dev/releases/vllm-v0-33-0/index.md)
- [vLLM v0.33.0 CUDA 13 / speculative – labmemo](https://labmemo.com/vllm-v0-33-0-cuda13-speculative-2026-11/)
- [vLLM v0.33.0 – remio.ai](https://www.remio.ai/post/vllm-v0-33-0-expands-model-support-and-ends-cuda-12-9-release-builds)

**llm-d / GIE / RHOAI**
- [RHOAI 3.5 – Configuring scheduler settings for LLM inference services](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/deploy_models_using_distributed_inference_with_llm-d/configuring-llm-scheduler)
- [RHOAI 3.5 – Deploying models by using Distributed Inference with llm-d](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/deploy_models_using_distributed_inference_with_llm-d/deploying-models-using-distributed-inference_distributed-inference)
- [llm-d – Precise Prefix Cache Routing (guide)](https://llm-d.ai/docs/dev/guides/precise-prefix-cache-routing)
- [llm-d – Precise prefix cache routing (well-lit path)](https://llm-d.ai/docs/well-lit-paths/precise-prefix-cache-routing)
- [llm-d – Prefix-Cache Aware Routing (architecture)](https://llm-d.ai/docs/0.7/architecture/advanced/kv-management/prefix-cache-aware-routing)
- [Gateway API Inference Extension – Prefix-aware EPP configuration](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/prefix-aware/)
- [GIE PR #1888 – autoTune du prefix cache scorer](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1888)

**Données de charge**
- [GitHub Copilot coding agent traces – Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/08/ghcp_traces-6.pdf)
- [TraceLab: Characterizing Coding Agent Workloads for LLM Serving – UW](https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/)
- [State of AI: An Empirical 100 Trillion Token Study with OpenRouter (arXiv 2601.10088)](https://arxiv.org/html/2601.10088v1)
- [OpenRouter – State of AI](https://openrouter.ai/state-of-ai)
