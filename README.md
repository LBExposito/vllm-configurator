# vllm-configurator

Application web qui génère une configuration `vllm serve` adaptée à votre déploiement : GPU, modèle, version de
vLLM, profil de charge et objectifs de latence. Elle explique chaque valeur retenue et signale ce qui reste à
vérifier.

- [Ce que fait l'app](#ce-que-fait-lapp)
- [Guide de démarrage](#guide-de-démarrage)
- [Maintenance : architecture et catalogue](#maintenance--architecture-et-catalogue)

---

## Ce que fait l'app

### Le problème

Les paramètres de vLLM qui pèsent le plus sur les performances (`--max-num-batched-tokens`, `--max-num-seqs`,
`--long-prefill-token-threshold`, speculative decoding, capture des CUDA graphs, `--gpu-memory-utilization`)
dépendent les uns des autres, du profil de charge et de la version de vLLM. Leurs règles de dimensionnement
existent, mais elles sont dispersées entre des PR, des notes de version et des recettes, et elles changent d'une
version à l'autre. L'app applique ces règles de façon reproductible et montre son raisonnement.

### Entrées

| Groupe | Ce que vous fournissez |
|---|---|
| Matériel | Type de GPU, nombre de GPU par instance (tensor parallel), nombre d'instances |
| Modèle et version | Modèle du catalogue, version de vLLM |
| Profil de charge | Type d'usage (chat, agentique, RAG, extraction, résumé, scoring, batch), mode interactif ou batch, ISL et OSL en P50 / P95 / P99, taux de prefix cache hit attendu, concurrence cible |
| SLO | TTFT p95 et ITL p95 cibles (optionnels) |
| Mesures en production (optionnel) | ITL p95 et TTFT p95 mesurés avec la config générée : ils priment sur l'estimation, et l'ITL recalibre le modèle de latence |
| Speculative decoding | Activation, `k`, méthode d'échantillonnage du draft, longueur d'acceptation mesurée, tolérance à une vérification lossy (FLy) |
| Logs de démarrage (optionnel) | `GPU KV cache size`, taille de bloc et `gpu-memory-utilization` lus dans les logs d'une instance |

Seuls l'ISL P50 et l'OSL P50 sont obligatoires. Une donnée manquante est remplacée par une hypothèse affichée, ou
l'app la demande explicitement si elle est indispensable.

### Sorties

- **La configuration**, en deux formats : le bloc `args:` à coller dans un manifeste KServe ou Kubernetes, et la
  commande `vllm serve` équivalente.
- **Une justification pour chaque valeur** : la règle appliquée, la formule et le calcul chiffré, la valeur par
  défaut de vLLM, les sources, et un statut ✅ vérifié ou 🔶 à vérifier.
- **Des avertissements** classés par gravité : modèle non pris en charge par la version, FLy lossy en usage
  agentique, OSL suspect, KV limitant, données non confirmées, changements cassants de la version.
- **Des indicateurs calculés** : ratio ISL_eff/OSL réel comparé au ratio nécessaire pour remplir le budget,
  contextes que le KV peut loger aux P50/P95/P99, tokens consommés par step de decode, tokens par expert MoE,
  plafond de séquences imposé par vLLM avec DSpark.
- **L'interprétation SLO** : un verdict (✅ OK, ⚠️ à risque, 🛑 KO), le goulot d'étranglement (prefill, KV cache,
  places de séquence, decode, interférence du prefill sur le decode, plancher d'ITL) et les corrections chiffrées :
  réglage de paramètres à capacité constante, nombre d'instances à ajouter, passage à un TP plus large, ou piste
  d'architecture (prefix cache, désagrégation P/D, DP-attention…). Voir [l'interprétation SLO](#linterprétation-slo).
- **Les hypothèses** retenues quand une entrée manque.
- **La vérification** : une commande pour contrôler que chaque flag existe sur l'image déployée, et la liste des
  métriques Prometheus à surveiller.

### Ce sur quoi vous pouvez compter

- **Aucun flag inconnu.** Chaque flag et chaque clé JSON émis ont été extraits du code source de vLLM au tag de
  la version choisie. C'est important à partir de 0.33, dont le frontend Rust refuse de démarrer sur un flag
  inconnu.
- **Des règles qui suivent la version.** Par exemple, avec DSpark à k = 3, chaque séquence en decode consomme
  6 tokens de budget en 0.32 et 4 en 0.33. FLy n'est proposé qu'à partir de 0.33.
- **Aucune spécification inventée.** Les fiches modèles sont épinglées sur une révision Hugging Face précise.
  Quand la taille du KV par token n'est pas calculable de façon fiable, l'app demande la valeur des logs au lieu
  de l'estimer.

### Limites

- L'app ne mesure rien : elle applique des règles. L'ITL et le TTFT de l'interprétation SLO sont des **ordres de
  grandeur** issus d'un modèle roofline dont les rendements sont supposés (🔶). Saisissez l'ITL p95 mesurée après
  déploiement pour recaler le modèle.
- Sans logs de démarrage, la taille du KV est une estimation grossière (borne haute) : relancez le calcul avec la
  valeur réelle après le premier déploiement.
- La configuration de llm-d (EPP) n'est pas encore générée. Elle est prévue pour une v2, qui réutilisera la taille
  du KV et la taille de bloc calculées ici.

### Catalogue actuel

| Type | Contenu |
|---|---|
| GPU | NVIDIA B300 |
| Modèles | DeepSeek V4.1 Flash (build 1017), Qwen3.8 Flash Next, GLM-5.3 Flash, Mistral Large 4 (image dédiée requise) |
| Versions de vLLM | 0.31.0, 0.32.0, 0.33.0 |

### Exemple : le cas de référence

DeepSeek V4.1 Flash sur 2 pods de 4 × B300 (TP4), vLLM 0.33.0, chat agentique, ISL P50 24K, KV de 86 M tokens par
pod lu dans les logs. Configuration générée :

```yaml
args:
  # --- Parallélisme ---
  - --tensor-parallel-size
  - "4"
  - --data-parallel-size
  - "1"

  # --- Scheduler ---
  - --max-num-seqs
  - "512"
  - --max-num-batched-tokens
  - "8192"
  - --long-prefill-token-threshold
  - "2048"
  - --long-prefill-token-threshold-adaptive
  - --enable-chunked-prefill

  # --- KV cache / prefix caching ---
  - --enable-prefix-caching
  - --kv-cache-dtype
  - "fp8"
  - --gpu-memory-utilization
  - "0.8"
  # pas de block-size : imposée à 128 par le backend, ne pas la forcer (§5)
  # pas de kv-offloading-size : 3 592 contextes P50 tiennent dans le KV, offload inutile
  # pas de max-model-len : défaut = contexte maximal du modèle (1 048 576)

  # --- Speculative decoding ---
  - --speculative-config
  - '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false}'

  # --- CUDA graphs ---
  - --compilation-config
  - '{"max_cudagraph_capture_size":2048}'
```

Pour ce cas, l'app signale aussi que l'OSL P50 de 4 000 tokens est environ 16 fois la médiane publiée pour
l'agentique (247), et que l'ISL P99 de 666K est à qualifier : mesure par appel ou cumul par session ?

---

## Guide de démarrage

### 1. Prérequis

- Python 3.11 ou plus récent.
- [uv](https://docs.astral.sh/uv/) pour l'environnement et les dépendances.
- Aucun GPU n'est nécessaire : l'app ne lance pas vLLM.

### 2. Installation

Depuis la racine du projet :

```bash
uv sync
```

uv crée un environnement `.venv` et y installe le projet ainsi que ses dépendances (Streamlit, pydantic, PyYAML,
pytest).

### 3. Lancer l'interface

```bash
uv run streamlit run src/vllm_configurator/web/app.py
```

L'app s'ouvre sur <http://localhost:8501>. Le formulaire est dans la barre latérale et les résultats dans la
partie centrale.

### 4. Premier essai : le cas de référence

Cliquez sur **« Charger le cas de référence (§1) »** en haut de la barre latérale. Le formulaire se remplit avec
le déploiement réel décrit dans `docs/vllm-config-knowledge-base.md`, et la configuration s'affiche aussitôt.
Parcourez les onglets pour voir comment chaque valeur est justifiée.

### 5. Décrire votre propre déploiement

Remplissez la barre latérale de haut en bas. Les résultats se recalculent à chaque modification.

| Section | Conseils |
|---|---|
| **Matériel** | « GPU par instance » est le tensor parallel d'un pod ; « Instances » est le nombre de pods indépendants. |
| **Modèle et version** | Choisissez la version de vLLM réellement déployée : plusieurs règles en dépendent. |
| **Profil de charge** | Mesurez ISL et OSL **par appel au modèle**, pas par tour de conversation ni par session. Laissez le hit rate vide si vous ne l'avez pas mesuré : l'app prend la valeur publiée pour votre type d'usage et l'affiche dans « Hypothèses ». La concurrence cible est un total sur toutes les instances. |
| **SLO** | Optionnels. Ils ne changent pas la config générée ; ils alimentent l'interprétation SLO (verdict, goulot, correction). Renseignez aussi la concurrence cible : sans elle, l'analyse suppose des instances pleines et ne peut pas chiffrer les GPU à ajouter. |
| **Mesures en production** | ITL p95 et TTFT p95 relevés avec la config générée. La mesure tranche le verdict ; l'ITL recalibre toutes les durées estimées. |
| **Speculative decoding** | Laissez `k` vide pour utiliser la valeur recommandée pour le modèle. Ne cochez « Tolérer une vérification lossy » qu'en A/B : FLy peut modifier les sorties, ce qui est risqué pour les tool calls. |
| **Logs de démarrage** | Si une instance tourne déjà, recopiez `GPU KV cache size: N tokens`, la taille de bloc (`Setting kv cache block size to N`) et la `gpu-memory-utilization` utilisée. C'est la donnée qui fiabilise le plus la configuration. |
| **Avancé** | Forcer `gpu-memory-utilization`, ou indiquer une réserve mémoire par GPU pour affiner l'estimation du KV sans logs. |

### 6. Lire les résultats

Le bandeau du haut résume `max-num-seqs`, le budget de tokens, la taille du KV et le nombre d'avertissements. Les
avertissements critiques s'affichent juste en dessous.

| Onglet | Contenu |
|---|---|
| **Configuration** | Bloc `args:` (téléchargeable en `vllm-args.yaml`) et commande `vllm serve`. |
| **Interprétation SLO** | Verdict, contrôles (KV, charge de prefill, places, ITL, TTFT), goulots, recommandations avec les flags à changer, décomposition du step. |
| **Justifications** | Une fiche par paramètre : valeur, émis ou non, règle, formule, calcul, défaut de vLLM, sources, statut. |
| **Avertissements** | Classés en critique / avertissement / info. Les points 🔶 « À vérifier » viennent de données non confirmées. |
| **Indicateurs** | Valeurs calculées et leur lecture. Le plus utile : le ratio ISL_eff/OSL réel comparé au ratio nécessaire pour remplir le budget de tokens. |
| **Hypothèses** | Ce que l'app a supposé faute de donnée. Fournir la donnée réelle supprime l'hypothèse. |
| **Vérification et métriques** | Commande de vérification des flags, métriques à surveiller, test de validation du seuil de prefill. |

### 7. Déployer la configuration

1. **Vérifiez les flags sur l'image réellement déployée**, avec la commande de l'onglet « Vérification et
   métriques ». Elle liste les flags que l'image reconnaît ; chaque flag de la config doit y apparaître :
   ```bash
   vllm serve --help=all | grep -oE -- "--(no-)?(max-num-seqs|max-num-batched-tokens|...)\b" | sort -u
   ```
2. **Collez le bloc `args:`** dans la définition du conteneur vLLM de votre manifeste. Ce bloc ne contient pas le
   nom du modèle, généralement fourni ailleurs par KServe. La commande `vllm serve`, elle, l'inclut.
3. **Déployez**, puis vérifiez dans les logs que vLLM démarre sans erreur ni OOM pendant la capture des CUDA graphs.

### 8. Après le premier déploiement : boucler

La taille du KV dépend de `gpu-memory-utilization`, de `max-num-seqs` et de la capture des CUDA graphs. Elle
change donc avec la nouvelle configuration.

1. Relevez dans les logs la nouvelle ligne `GPU KV cache size` et la taille de bloc.
2. Saisissez-les dans « Logs de démarrage » : l'app recalcule `max-num-seqs` et les indicateurs.
3. Surveillez les métriques listées, en particulier l'ITL p95 et le hit rate du prefix cache. Si l'ITL dépasse la
   cible, appliquez la consigne de repli donnée dans les avertissements (par exemple 384 séquences et une capture
   de 1536).
4. Mesurez la longueur d'acceptation `a = 1 + spec_decode_num_accepted_tokens / spec_decode_num_drafts` et
   saisissez-la : elle remplace l'hypothèse a = 3.

### L'interprétation SLO

L'analyse (`src/vllm_configurator/analysis.py`) évalue le point de fonctionnement de la config générée, puis cherche
comment le corriger.

**Contrôles de capacité**, sans modèle matériel :

| Contrôle | Calcul | Seuil |
|---|---|---|
| Occupation du KV | S × (ISL P50 + OSL P50 / 2) / KV par instance | 90 % (au-delà : préemptions) |
| Charge de prefill du budget | prefill non caché demandé par step (S·a/OSL × ISL_eff) / (B − S·d), en régime stable (§3.1) | 80 % (au-delà : file d'attente avant prefill) |
| Places de séquence | concurrence cible par instance vs `max-num-seqs` | file d'attente comptée dans le TTFT |

**Latences**, par un modèle roofline d'un step du scheduler (`src/vllm_configurator/perf.py`) : le step dure le maximum
entre la lecture mémoire (poids non routés, experts MoE touchés par au moins un token, KV des séquences en decode)
et le calcul (2 × paramètres activés par token), plus un surcoût fixe par couche. L'**ITL p95** est la durée d'un
step chargé en prefill ; le **TTFT p95** additionne l'attente d'une place, l'attente du prefill (file M/D/1) et le
prefill du prompt P95 non caché, découpé selon le seuil de long prefill. Une reprise à froid (ISL P95 sans cache)
est affichée hors verdict.

**Verdict.** Une estimation sous 85 % de la cible est ✅, entre 85 et 115 % ⚠️, au-delà 🛑. Une mesure fournie est
comparée strictement à la cible.

**Corrections**, dans l'ordre affiché :

1. Le réglage de `--max-num-seqs`, `--max-num-batched-tokens` et `--long-prefill-token-threshold` qui tient tous les
   contrôles avec le moins de changements, à capacité GPU constante (la capture des CUDA graphs suit).
2. Le nombre minimal d'instances (avec leur réglage) si la concurrence cible est fournie.
3. Le passage à un TP deux fois plus large (jusqu'à 8), quand il coûte moins de GPU : c'est le levier quand l'ITL
   est bornée par la lecture des poids, ce qui est fréquent avec les MoE.
4. Les pistes propres au goulot : `gpu-memory-utilization`, DP-attention, offload du KV, prefix cache hit rate,
   désagrégation prefill/decode, PCP.

Les rendements du modèle (`perf_hbm_efficiency`, `perf_mfu`, `perf_overhead_per_layer_ms`) et les seuils sont des
heuristiques de `catalog/workload_profiles.yaml`, toutes 🔶. Le modèle exige le bloc `performance` de la fiche
modèle et les blocs `bandwidth` et `compute` de la fiche GPU ; sans eux, seuls les contrôles de capacité sont faits.

### 9. Dépannage

| Symptôme | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'vllm_configurator'` | `uv sync --reinstall-package vllm-configurator` |
| « Taille du KV inconnue » | Le modèle a une architecture dont le KV ne se calcule pas de façon fiable : saisissez `GPU KV cache size` lu dans les logs. |
| Avertissement critique « absente du registre de vLLM » | La version choisie ne prend pas en charge ce modèle. Utilisez l'image dédiée indiquée, puis vérifiez les flags dessus. |
| Modèle, GPU ou version absent des listes | Ajoutez-le au catalogue (voir ci-dessous). |
| Caractères accentués illisibles dans un terminal Windows | Préfixez les scripts par `PYTHONUTF8=1`. |

### 10. Lancer les tests

```bash
uv run pytest
```

Les tests couvrent les formules (avec les valeurs numériques de la base de connaissances), les règles et leur
dépendance à la version, la cohérence du catalogue, le golden test et l'interface.

---

## Maintenance : architecture et catalogue

La base de connaissances d'origine est `docs/vllm-config-knowledge-base.md`. Les vérifications faites dans le code
source de vLLM, dont plusieurs corrections à cette base, sont dans `docs/verification-vllm-0.33.0.md`.

### Architecture

```
catalog/                    Données versionnées (YAML), séparées de la logique
  gpus/<id>.yaml            GPU : HBM, famille de compute capability…
  models/<id>.yaml          Modèles : config.json épinglé, KV, taille de bloc, spec decoding natif, support vLLM
  vllm/<version>.yaml       Versions de vLLM : défauts, comportements, flags émis, changements cassants
  vllm/extracted/*.yaml     Flags et clés JSON extraits du code source au tag (générés)
  workload_profiles.yaml    Profils de charge, heuristiques, métriques
src/vllm_configurator/
  catalog/                  Schémas pydantic et chargement + validation croisée
  formulas.py               Formules de la §3, fonctions pures
  perf.py                   Modèle roofline d'un step du scheduler (latence)
  analysis.py               Interprétation SLO : verdict, goulot, recherche de la correction
  rules/                    Règles : une fonction pure (RuleContext, state) -> RuleOutput par décision
  engine.py                 Exécute le pipeline de règles, produit un GenerationResult
  render.py                 YAML `args` (KServe/K8s), ligne de commande, commande de vérification
  targets/                  Registre des cibles : `vllm` en v1, `llmd_epp` prévu en v2
  web/app.py                Interface Streamlit
tests/
  unit/                     Formules, règles, catalogue, UI
  golden/reference_case/    Cas de référence : entrées §1, config attendue §7, écarts justifiés
tools/extract_vllm_flags.py Extraction des flags d'une version de vLLM depuis GitHub
```

Principes :

- **Les règles ne codent aucune version en dur.** Elles lisent les comportements du catalogue de la version
  (`ctx.behavior("spec_decode.extra_draft_slots")`, etc.). Changer de version, c'est changer de fichier.
- **Aucun flag inconnu n'est émis.** Le rendu refuse un flag absent de `catalog/vllm/extracted/<version>.yaml`. Le
  moteur refuse aussi une clé `--speculative-config` inconnue.
- **Honnêteté sur l'incertitude.** Toute donnée `unverified` utilisée par une règle produit un avertissement
  « À vérifier ». Une donnée manquante produit soit une hypothèse affichée, soit une erreur bloquante qui demande
  la valeur.
- **Toujours émettre.** Les flags sur lesquels repose la config sont émis même quand ils valent le défaut. Font
  exception `--block-size` (ne jamais forcer une taille imposée par le backend), `--kv-offloading-size` (sa seule
  présence active l'offload) et `--max-model-len` (défaut = contexte maximal du modèle). Leur raison figure en
  commentaire dans le YAML.
- **Préparation de la v2 (llm-d).** `GenerationResult.facts` (`DeploymentFacts`) expose le KV par instance et sa
  source, la taille de bloc, `max_num_seqs`, etc. Le futur target `llmd_epp` consommera ces faits
  (`lruCapacityPerServer = formulas.lru_capacity_per_server(kv, block)`) sans toucher au moteur vLLM.

### Ajouter un GPU

Le skill Claude Code `vllm-configurator-catalog` (`.claude/skills/vllm-configurator-catalog/`) automatise cette procédure et
celle des modèles. À la main :

1. Créer `catalog/gpus/<id>.yaml` (le nom du fichier doit être égal à `id`), sur le modèle de `b300.yaml`.
2. `memory.hbm_gb` en Go décimaux, d'après la fiche constructeur. Si seule une page système donne un total,
   diviser, mais marquer `status: unverified` et expliquer le recoupement dans `note`.
3. `cudagraph_family` : `blackwell_dc` pour une compute capability 10.x, sinon `other`. C'est la seule
   information que vLLM utilise pour le plafond de capture des CUDA graphs.
4. `bandwidth.hbm_tb_s` et `compute.dense_tflops` (fp4, fp8, bf16, valeurs **denses** par GPU) : facultatifs, ils
   servent au modèle de latence de l'interprétation SLO.
5. Valider (voir plus bas).

La HBM ne sert qu'à estimer le KV quand les logs de démarrage ne sont pas fournis.

### Ajouter un modèle

1. Identifier le dépôt Hugging Face exact et collecter les faits à une révision épinglée :
   ```bash
   PYTHONUTF8=1 uv run python .claude/skills/vllm-configurator-catalog/scripts/hf_model_facts.py <org/repo>
   ```
2. Lire la recette `https://recipes.vllm.ai/<org/repo>` : méthode et `k` de spec decoding, dtype du KV, version
   minimale, image dédiée.
3. Créer `catalog/models/<id>.yaml` sur le modèle des fiches existantes. Règles :
   - `hf_revision` : SHA complet de 40 caractères ; les URLs `source` pointent vers cette révision.
   - `kv_cache.bytes_per_token` : laisser `null` si l'architecture est exotique (attention linéaire ou hybride,
     compression, KV partagé, indexeurs). L'app exigera alors la ligne « GPU KV cache size » des logs, ce qui est
     voulu. Une formule standard (GQA, MLA simple) se met dans `method`, avec `status: unverified`.
   - `block_size.forced_by_backend` : seulement d'après un log réel ou une consigne explicite de la recette.
   - `speculative.native_method` doit figurer dans `speculative_methods` du fichier `extracted` de la version visée.
   - `performance` (facultatif, pour l'interprétation SLO) : paramètres activés par token d'après la model card ou
     la recette, taille des experts routés d'après `config.json`, octets par paramètre selon la quantization.
4. Valider.

Le détail champ par champ est dans `.claude/skills/vllm-configurator-catalog/references/fields.md`.

### Ajouter une version de vLLM

1. **Extraire les flags** au tag de la version :
   ```bash
   uv run python tools/extract_vllm_flags.py v0.34.0
   ```
   L'outil génère `catalog/vllm/extracted/0.34.0.yaml` : flags CLI, clés de `--speculative-config` et de
   `--compilation-config`, méthodes de spec decoding, architectures du registre. Les fichiers sources sont
   téléchargés dans `.cache/vllm-src/` et lus comme du texte, jamais exécutés.
2. **Copier le catalogue précédent** (`catalog/vllm/0.33.0.yaml` → `0.34.0.yaml`), puis mettre à jour `version`,
   `source_tag` et `verified_on`.
3. **Revérifier chaque comportement dans le code source au nouveau tag** et corriger `value`, `source` et
   `status`. Où regarder :

   | Comportement | Fichier, symbole |
   |---|---|
   | `defaults.*` | `vllm/config/cache.py`, `scheduler.py`, `speculative.py` (valeurs des champs) |
   | `long_prefill.*` | `vllm/v1/core/sched/scheduler.py`, `schedule()` (`num_eligible_reqs`, `adaptive_long_prefill_threshold`) |
   | `spec_decode.verification_tokens_in_budget` | `scheduler.py`, calcul de `num_new_tokens` pour les requêtes running |
   | `spec_decode.extra_draft_slots` | `vllm/config/speculative.py`, `max_num_new_slots_for_drafting` (tableau dans la docstring) |
   | `spec_decode.dspark_active_seq_cap` | `vllm/config/vllm.py`, `max_draft_seqs` |
   | `spec_decode.fly`, `adaptive_verification`, `draft_lm_head_quantization` | `vllm/config/speculative.py` |
   | `cudagraph.*` | `vllm/config/vllm.py`, `_set_cudagraph_sizes` ; `vllm/config/compilation.py` |

   Ce qui n'a pas pu être relu dans le code reste en `status: unverified`, avec la source utilisée (notes de
   version…). L'app l'affichera « à vérifier ».
4. **Mettre à jour** `breaking_changes` et `notes` d'après les notes de version.
5. Si une règle a besoin d'un comportement nouveau, ajouter la clé dans **tous** les catalogues de versions, avec la
   valeur propre à chacune, puis la lire dans la règle via `ctx.behavior(...)`. `tests/unit/test_catalog.py`
   vérifie que chaque version définit les comportements utilisés.
6. Valider.

### Valider le catalogue

```bash
uv run python -c "from vllm_configurator.catalog import load_catalog; c = load_catalog(); print(c.validate() or 'OK')"
uv run pytest
```

### Golden test

`tests/golden/reference_case/` contient les entrées du §1 (`input.yaml`), la config du §7 recopiée telle quelle
(`expected.yaml`) et les écarts assumés (`deviations.yaml`). Le test échoue si la config générée diffère de la §7
sans que l'écart soit listé et justifié, ou si un écart listé ne se produit plus. Écarts actuels :

- flags ajoutés par la politique « toujours émettre » (`--tensor-parallel-size 4`, `--data-parallel-size 1`,
  `--enable-chunked-prefill`, clés `rejection_sample_method` et `enable_adaptive_verification`) ;
- `--long-prefill-token-threshold-adaptive`, sans lequel le comportement dynamique du §2.5 n'est pas actif en 0.33 ;
- même budget de 8192, mais justification corrigée : avec 1 + k tokens de vérification par séquence, le ratio
  nécessaire est de 4,0, et 8192 est retenu pour l'ITL en interactif.
