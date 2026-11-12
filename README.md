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
| Profil de charge | Type d'usage (chat, agentique, RAG, extraction, résumé, scoring, batch), mode interactif ou batch, ISL et OSL en P50 / P95 / P99, taux de prefix cache hit attendu, requêtes en vol visées |
| SLO | TTFT p95 et ITL p95 cibles (optionnels) |
| Mesures en production (optionnel) | ITL (ou TPOT) p95 et TTFT p95 mesurés avec la config générée : ils priment sur l'estimation, et la latence de génération recale le modèle |
| Speculative decoding | Activation, `k`, méthode d'échantillonnage du draft, longueur d'acceptation mesurée, tolérance à une vérification lossy (FLy) |
| Contexte et arguments supplémentaires (optionnel) | `max-model-len` cible (avec extension de contexte YaRN quand le modèle en a une), flags vLLM et variables d'environnement additionnels |
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
- **Un plan de test par paliers** : pour chaque valeur de `max-num-seqs` à valider, les flags à poser, la charge à
  injecter, les plages attendues d'ITL, de TPOT, de TTFT et d'occupation du KV, et le nombre d'instances nécessaire si
  le palier tient. La mesure des paliers fixe le dimensionnement ; l'estimation sert à choisir où tester. Une
  estimation indicative au point de la config générée (goulot, pistes de correction) reste disponible. Voir
  [Paliers de test et SLO](#paliers-de-test-et-slo).
- **Les hypothèses** retenues quand une entrée manque.
- **La vérification** : une commande pour contrôler que chaque flag existe sur l'image déployée, et la liste des
  métriques Prometheus à surveiller.
- **Un fichier de scénario** YAML, exportable et réimportable depuis l'interface : entrées, SLO et mesures, plus un
  instantané de la config générée. Voir [Sauvegarder et recharger un scénario](#sauvegarder-et-recharger-un-scénario).

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

- L'app ne mesure rien : elle applique des règles. Les latences des paliers sont des **plages** (×0,8 à ×2,5 sans
  mesure) issues d'un modèle roofline dont les rendements sont supposés (🔶). C'est la mesure des paliers qui tranche ;
  saisir la latence mesurée recale le modèle (±15 %).
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
| **Profil de charge** | Mesurez ISL et OSL **par appel au modèle**, pas par tour de conversation ni par session. Laissez le hit rate vide si vous ne l'avez pas mesuré : l'app prend la valeur publiée pour votre type d'usage et l'affiche dans « Hypothèses ». Les requêtes en vol visées sont un total sur toutes les instances : requêtes en cours + en file, **pas des utilisateurs** (un utilisateur agentique n'a une requête en cours qu'une partie du temps). Mesure : `vllm:num_requests_running` + `vllm:num_requests_waiting` sommés sur les pods. |
| **SLO** | Optionnels. Ils ne changent pas la config générée ; ils fixent les critères de passage des paliers. Choisissez sur quoi porte la cible de génération : **ITL** (`vllm:inter_token_latency_seconds`, intervalle entre deux sorties, soit un step) ou **TPOT** (`vllm:request_time_per_output_token_seconds`, temps par token). Avec le speculative decoding, l'ITL monte un peu mais le TPOT est divisé par ~a : pour le ressenti utilisateur, c'est le TPOT qui compte. |
| **Mesures en production** | Latence de génération (même métrique que la cible) et TTFT p95 relevés au palier de la config générée. La mesure tranche ; la latence de génération recale toutes les durées estimées. |
| **Speculative decoding** | Laissez `k` vide pour utiliser la valeur recommandée pour le modèle. Ne cochez « Tolérer une vérification lossy » qu'en A/B : FLy peut modifier les sorties, ce qui est risqué pour les tool calls. |
| **Logs de démarrage** | Si une instance tourne déjà, recopiez `GPU KV cache size: N tokens`, la taille de bloc (`Setting kv cache block size to N`) et la `gpu-memory-utilization` utilisée. C'est la donnée qui fiabilise le plus la configuration. |
| **Contexte et arguments supplémentaires** | Voir ci-dessous. |
| **Avancé** | Forcer `gpu-memory-utilization`, ou indiquer une réserve mémoire par GPU pour affiner l'estimation du KV sans logs. |

#### Étendre le contexte et ajouter des arguments vLLM

**`max-model-len` cible.** Vide, le modèle sert son contexte natif et `--max-model-len` n'est pas émis. En dessous du
natif, la valeur est émise telle quelle. Au-delà, l'app applique l'extension de contexte de la fiche modèle quand
elle en a une. Exemple avec Qwen3.8 Flash Next (natif 262 144, 1 000 000 via YaRN d'après la model card) :

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3.8-Flash-Next \
  --max-model-len 1000000 \
  --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"original_max_position_embeddings":262144,"factor":4.0}}}'
```

- `factor` est le plus petit multiple de 0,5 qui couvre `max-model-len / contexte natif`, soit 2,0 pour 524 288 et
  4,0 pour 1 000 000, comme dans la model card.
- Le bloc `rope_parameters` est passé en entier : vLLM le remplace en bloc, alors qu'il fusionne `text_config`.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` est indispensable de 0.31 à 0.33 : pour YaRN, vLLM n'applique pas le facteur au
  maximum dérivé de `config.json` et refuse sinon de démarrer. L'onglet Configuration affiche alors un bloc `env:` à
  poser sur le conteneur, et la commande `vllm serve` le préfixe.
- YaRN est statique : il s'applique à toutes les requêtes et peut dégrader les textes courts. N'étendez le contexte
  que si le trafic l'exige.

**Arguments vLLM supplémentaires.** Un dictionnaire YAML, une entrée par flag. Un `dict` est émis en JSON (une
chaîne JSON aussi), `true` émet le flag seul, `--no-<flag>: true` désactive un booléen.

```yaml
--enable-log-requests: true
--served-model-name: qwen-long
--hf-overrides: {text_config: {rope_parameters: {rope_type: yarn, factor: 2.0}}}
```

- Chaque flag est vérifié contre `catalog/vllm/extracted/<version>.yaml`. Un flag inconnu est ignoré, avec un
  avertissement critique : le frontend Rust de vLLM ≥ 0.33 refuserait de démarrer.
- Un flag que l'app génère déjà est remplacé par la valeur fournie. Pour `--max-num-seqs`, `--max-num-batched-tokens`,
  `--speculative-config`, `--tensor-parallel-size` et `--max-model-len`, un avertissement rappelle que l'analyse
  reste calculée sur la valeur générée.
- Un `--hf-overrides` fourni à la main remplace l'extension du catalogue. Avec un `max-model-len` au-delà du natif,
  l'app pose quand même `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

**Variables d'environnement.** Un dictionnaire YAML (`VLLM_LOGGING_LEVEL: DEBUG`), ajouté au bloc `env:`.

Le contexte, les arguments et les variables sont enregistrés dans les fichiers de scénario.

#### Sauvegarder et recharger un scénario

Le panneau **« 💾 Scénarios »** en haut de la barre latérale exporte le formulaire complet (matériel, modèle, profil
de charge, SLO, mesures, logs de démarrage) dans un fichier YAML, et réimporte un tel fichier pour remplir le
formulaire. Donnez un nom au scénario avant l'export : il sert de nom de fichier.

```yaml
kind: vllm-configurator/scenario
version: 1
name: Prod agentique B300
saved_at: '2026-11-12T10:00:00'
request:                  # seule partie relue à l'import
  gpu_id: b300
  gpus_per_instance: 4
  num_instances: 2
  model_id: deepseek-v4.1-flash-1017
  vllm_version: 0.33.0
  workload: {usage_type: agentic, mode: interactive, isl: {p50: 24000}, osl: {p50: 250}, target_concurrency: 800}
  slo: {ttft_p95_ms: 2000.0, itl_p95_ms: 40.0}
generated:                # instantané informatif (args, verdict SLO), ignoré et recalculé à l'import
  args: [...]
```

Les fichiers peuvent être versionnés à côté des manifestes. L'import accepte aussi une requête nue, au format de
`tests/golden/reference_case/input.yaml`. Il refuse un fichier dont le GPU, le modèle ou la version de vLLM manque
au catalogue, et indique pourquoi.

### 6. Lire les résultats

Le bandeau du haut résume `max-num-seqs`, le budget de tokens, la taille du KV et le nombre d'avertissements. Les
avertissements critiques s'affichent juste en dessous.

| Onglet | Contenu |
|---|---|
| **Configuration** | Bloc `args:` (téléchargeable en `vllm-args.yaml`) et commande `vllm serve`. |
| **Paliers et SLO** | Plan de test par paliers (flags, charge, plages attendues, instances si validé), protocole et critères de passage, contrôles au point de la config générée, estimation indicative (goulots, pistes). |
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
3. Déroulez le plan de paliers de l'onglet « Paliers et SLO » et saisissez la latence mesurée au palier de la config
   générée : les plages des autres paliers se resserrent. Surveillez aussi le hit rate du prefix cache.
4. Mesurez la longueur d'acceptation `a = 1 + spec_decode_num_accepted_tokens / spec_decode_num_drafts` et
   saisissez-la : elle remplace l'hypothèse a = 3.

### Paliers de test et SLO

Un modèle de latence non mesuré peut se tromper d'un facteur 2 ou 3. L'app ne promet donc pas de verdict : elle donne
un **plan de paliers** qui encadre la limite, et c'est la mesure qui dimensionne. Le code est dans
`src/vllm_configurator/analysis.py`.

**Les paliers.** Ce sont des valeurs de `max-num-seqs` (16, 32, 64… 2048), plafonnées par ce que le KV loge et par les
requêtes en vol visées par instance. Pour chacune, l'app donne :

- les flags à poser : `--max-num-seqs`, la capture des CUDA graphs et, si la charge de prefill l'exige, un budget
  `--max-num-batched-tokens` plus haut ;
- la charge à injecter, autant de requêtes en vol que de séquences ;
- les plages attendues d'ITL, de TPOT, de TTFT, de débit et d'occupation du KV ;
- un pronostic : ✅ si même le cas pessimiste tient les SLO, 🛑 si même le cas optimiste les manque, ⚠️ entre les deux ;
- le nombre d'instances nécessaire si ce palier est le dernier validé : ⌈requêtes en vol visées / max-num-seqs⌉.

L'app retient les paliers autour de la limite attendue : deux paliers sûrs, la zone incertaine et le premier palier
en échec probable. Le palier de la config générée est marqué ★.

**Protocole.** Chaque palier se valide sur une instance, en boucle fermée et par rejeu de traces réelles (mêmes ISL,
OSL et thinking qu'en production). Le régime stable doit durer au moins trois fois la durée d'une requête. Un palier
passe si la latence de génération et le TTFT p95 tiennent la cible, sans file d'attente
(`vllm:num_requests_waiting` ≈ 0), sans préemption (`vllm:num_preemptions`) et avec `vllm:kv_cache_usage_perc` sous
90 %. Le premier palier en échec fixe la capacité d'une instance au palier précédent.

**Le modèle d'estimation**, qui sert à placer les paliers :

| Grandeur | Calcul |
|---|---|
| Longueurs | Moyennes d'ISL et d'OSL déduites des percentiles (lognormale ajustée sur P50/P95, corrigée si le P99 révèle une queue plus lourde), pas du P50 |
| Occupation du KV | S × (ISL moyen + OSL moyen / 2) / KV par instance, KV ramené au `gpu-memory-utilization` de la config si les logs ont été lus à une autre valeur |
| Charge de prefill | prefill non caché demandé par step (S·a/OSL × ISL_eff) / (B − S·d), en régime stable (§3.1) |
| Step | roofline (`src/vllm_configurator/perf.py`) : max(lecture des poids, experts MoE touchés et KV ; calcul) + surcoût fixe par couche |
| ITL / TPOT | ITL = step chargé en prefill ; TPOT = step moyen / a |
| TTFT | attente d'une place + attente du prefill (file M/D/1) + prefill du prompt P95 non caché, découpé selon le seuil de long prefill |

Sans mesure, les durées sont données de ×0,8 à ×2,5 ; recalées sur une latence mesurée, à ±15 %
(`perf_uncertainty`). Les rendements (`perf_hbm_efficiency`, `perf_mfu`, `perf_overhead_per_layer_ms`) et les seuils
sont des heuristiques de `catalog/workload_profiles.yaml`, toutes 🔶. Sans bloc `performance` dans la fiche modèle,
ni `bandwidth` et `compute` dans la fiche GPU, les paliers n'ont pas de plages de latence : seule la capacité est
contrôlée.

**L'estimation indicative**, repliée sous le plan, garde l'analyse au point de la config générée. Elle donne les
goulots (prefill, KV, places de séquence, decode, interférence du prefill, plancher d'ITL) et des pistes : réglage à
capacité constante, instances, TP plus large, `gpu-memory-utilization`, DP-attention, prefix cache, désagrégation
P/D, PCP. Une piste de réglage n'est proposée que si elle tient même dans le cas pessimiste ; elle reste à valider par
les paliers.

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
  analysis.py               Paliers de test, contrôles SLO, goulots et pistes de correction
  scenario.py               Fichiers de scénario YAML (export / import du formulaire)
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
   servent au modèle de latence des paliers de test.
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
   - `context_extension` (facultatif) : seulement d'après la model card ou la recette. `native_max_model_len`,
     `max_model_len` atteignable, `override_path` (ex. `[text_config, rope_parameters]`) et le bloc `rope_parameters`
     complet **sans** `factor`, que l'app calcule.
   - `performance` (facultatif, pour les plages de latence des paliers) : paramètres activés par token d'après la model card ou
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
