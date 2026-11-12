# vllm-config

Générateur de configuration `vllm serve` à partir du matériel, du modèle, de la version de vLLM, du profil de
charge et des SLO. Chaque valeur produite est accompagnée de la règle ou de la formule appliquée, de ses sources
et de son statut (✅ vérifié ou 🔶 à vérifier).

La base de connaissances d'origine est `docs/vllm-config-knowledge-base.md`. Les vérifications faites dans le code
source de vLLM, dont plusieurs corrections à cette base, sont dans `docs/verification-vllm-0.33.0.md`.

## Démarrage

```bash
uv sync
uv run streamlit run src/vllm_config/web/app.py   # interface web
uv run pytest                                      # tests (formules, règles, catalogue, golden test, UI)
```

Dans l'interface, le bouton « Charger le cas de référence (§1) » préremplit le cas réel (DeepSeek V4.1 Flash,
2 pods TP4 sur B300, vLLM 0.33.0).

## Architecture

```
catalog/                    Données versionnées (YAML), séparées de la logique
  gpus/<id>.yaml            GPU : HBM, famille de compute capability…
  models/<id>.yaml          Modèles : config.json épinglé, KV, taille de bloc, spec decoding natif, support vLLM
  vllm/<version>.yaml       Versions de vLLM : défauts, comportements, flags émis, changements cassants
  vllm/extracted/*.yaml     Flags et clés JSON extraits du code source au tag (générés)
  workload_profiles.yaml    Profils de charge, heuristiques, métriques
src/vllm_config/
  catalog/                  Schémas pydantic et chargement + validation croisée
  formulas.py               Formules de la §3, fonctions pures
  rules/                    Règles : une fonction pure (RuleContext, state) -> RuleOutput par décision
  engine.py                 Exécute le pipeline de règles, produit un GenerationResult
  render.py                 YAML `args` (KServe/K8s), ligne de commande, commande de vérification
  targets/                  Registre des cibles : `vllm` en v1, `llmd_epp` prévu en v2
  web/app.py                Interface Streamlit
tests/
  unit/                     Formules (valeurs du document), règles, catalogue, UI
  golden/reference_case/    Cas de référence : entrées §1, config attendue §7, écarts justifiés
tools/extract_vllm_flags.py Extraction des flags d'une version de vLLM depuis GitHub
```

Principes :

- **Les règles ne codent aucune version en dur.** Elles lisent les comportements du catalogue de la version
  (`ctx.behavior("spec_decode.extra_draft_slots")`, etc.). Changer de version, c'est changer de fichier.
- **Aucun flag inconnu n'est émis.** Le rendu refuse un flag absent de `catalog/vllm/extracted/<version>.yaml`,
  car le frontend Rust de vLLM ≥ 0.33 échoue sur les flags inconnus. Le moteur refuse aussi une clé
  `--speculative-config` inconnue.
- **Honnêteté sur l'incertitude.** Toute donnée `unverified` utilisée par une règle produit un avertissement
  « À vérifier ». Une donnée manquante produit soit une hypothèse affichée (onglet « Hypothèses »), soit une
  erreur bloquante qui demande la valeur (par exemple la taille du KV des logs quand elle n'est pas dérivable).
- **Toujours émettre.** Les flags sur lesquels repose la config sont émis même quand ils valent le défaut. Font
  exception `--block-size` (ne jamais forcer une taille imposée par le backend), `--kv-offloading-size` (sa seule
  présence active l'offload) et `--max-model-len` (défaut = contexte maximal du modèle). Leur raison figure en
  commentaire dans le YAML.
- **Préparation de la v2 (llm-d).** `GenerationResult.facts` (`DeploymentFacts`) expose le KV par instance et sa
  source, la taille de bloc, `max_num_seqs`, etc. Le futur target `llmd_epp` consommera ces faits
  (`lruCapacityPerServer = formulas.lru_capacity_per_server(kv, block)`) sans toucher au moteur vLLM.

## Ajouter un GPU

Le skill Claude Code `vllm-config-catalog` (`.claude/skills/vllm-config-catalog/`) automatise cette procédure
et celle des modèles. À la main :

1. Créer `catalog/gpus/<id>.yaml` (le nom du fichier doit être égal à `id`), sur le modèle de `b300.yaml`.
2. `memory.hbm_gb` en Go décimaux, d'après la fiche constructeur. Si seule une page système donne un total,
   diviser, mais marquer `status: unverified` et expliquer le recoupement dans `note`.
3. `cudagraph_family` : `blackwell_dc` pour une compute capability 10.x, sinon `other`. C'est la seule
   information que vLLM utilise pour le plafond de capture des CUDA graphs.
4. Valider (voir plus bas).

La HBM ne sert qu'à estimer le KV quand les logs de démarrage ne sont pas fournis.

## Ajouter un modèle

1. Identifier le dépôt Hugging Face exact et collecter les faits à une révision épinglée :
   ```bash
   PYTHONUTF8=1 uv run python .claude/skills/vllm-config-catalog/scripts/hf_model_facts.py <org/repo>
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
4. Valider.

Le détail champ par champ est dans `.claude/skills/vllm-config-catalog/references/fields.md`.

## Ajouter une version de vLLM

1. **Extraire les flags** au tag de la version :
   ```bash
   uv run python tools/extract_vllm_flags.py v0.34.0
   ```
   L'outil génère `catalog/vllm/extracted/0.34.0.yaml` : flags CLI, clés de `--speculative-config` et de
   `--compilation-config`, méthodes de spec decoding, architectures du registre. Les fichiers sources sont
   téléchargés dans `.cache/vllm-src/` et lus comme du texte, jamais exécutés.
2. **Copier le catalogue précédent** (`catalog/vllm/0.33.0.yaml` → `0.34.0.yaml`), puis mettre à jour
   `version`, `source_tag` et `verified_on`.
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
5. Si une règle a besoin d'un comportement nouveau, ajouter la clé dans **tous** les catalogues de versions, avec
   la valeur propre à chacune, puis lire la clé dans la règle via `ctx.behavior(...)`. `tests/unit/test_catalog.py`
   vérifie que chaque version définit les comportements utilisés.
6. Valider.

## Valider le catalogue

```bash
uv run python -c "from vllm_config.catalog import load_catalog; c = load_catalog(); print(c.validate() or 'OK')"
uv run pytest
```

## Golden test

`tests/golden/reference_case/` contient les entrées du §1 (`input.yaml`), la config du §7 recopiée telle quelle
(`expected.yaml`) et les écarts assumés (`deviations.yaml`). Le test échoue si la config générée diffère de la §7
sans que l'écart soit listé et justifié, ou si un écart listé ne se produit plus. Écarts actuels :

- flags ajoutés par la politique « toujours émettre » (`--tensor-parallel-size 4`, `--data-parallel-size 1`,
  `--enable-chunked-prefill`, clés `rejection_sample_method` et `enable_adaptive_verification`) ;
- `--long-prefill-token-threshold-adaptive`, sans lequel le comportement dynamique du §2.5 n'est pas actif en 0.33 ;
- même budget de 8192, mais justification corrigée : avec 1 + k tokens de vérification par séquence, le ratio
  nécessaire est de 4,0, et 8192 est retenu pour l'ITL en interactif.
