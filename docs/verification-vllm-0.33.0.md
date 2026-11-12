# Vérification des flags et comportements de vLLM 0.33.0

Vérifié le 12/11/2026 dans le code source de vLLM aux tags `v0.31.0`, `v0.32.0` et `v0.33.0`
(<https://github.com/vllm-project/vllm>). Les listes exhaustives de flags et de clés JSON sont générées par
`tools/extract_vllm_flags.py` dans `catalog/vllm/extracted/`.

Fichiers relus : `vllm/engine/arg_utils.py`, `vllm/config/scheduler.py`, `vllm/config/cache.py`,
`vllm/config/compilation.py`, `vllm/config/speculative.py`, `vllm/config/vllm.py`,
`vllm/v1/core/sched/scheduler.py`, `vllm/model_executor/models/registry.py`, `vllm/utils/argparse_utils.py`.

## Points ouverts de la §11 de la base de connaissances

| # | Point | Résultat |
|---|---|---|
| 1 | Noms exacts des flags d'offload KV et de `max_cudagraph_capture_size` | ✅ **Confirmé.** `--kv-offloading-size` (Gio, total sur les rangs TP ; offload désactivé si absent) et `--kv-offloading-backend` (`native` ou `lmcache`, défaut `native`) existent. `max_cudagraph_capture_size` est une clé de `--compilation-config` ; il existe aussi un raccourci `--max-cudagraph-capture-size`. |
| 2 | Les séquences en decode entrent-elles dans le diviseur du seuil dynamique (#58459) ? | ✅ **Oui.** Le diviseur vaut `len(running) + len(waiting)`, et les requêtes en decode font partie de `running`. Deux corrections au §2.5 : **(a)** le comportement est **opt-in** (`--long-prefill-token-threshold-adaptive`, défaut `False`), **(b)** c'est un **plancher** : `seuil_effectif = max(seuil, max_num_batched_tokens // (running + waiting))`. Sous forte charge, c'est donc le seuil fixe qui s'applique. Comportement identique en 0.31, 0.32 et 0.33. |
| 3 | Activation et défaut de la vérification adaptative DSpark | ✅ **Résolu.** Clé `enable_adaptive_verification` de `--speculative-config`, défaut `False`, « Currently only supported for method="dspark" ». |
| 6 | Spécificités de DeepSeek V4.1 | 🔶 **Partiellement résolu.** D'après `config.json` (révision `08c8657`) : 384 experts routés, top-6, 1 expert partagé, 40 couches. La taille du KV par token reste non dérivable de `config.json` (compression par couche, KV partagé, indexeur) ; la recette vLLM cite 890 octets par token, valeur non confrontée à vLLM. |
| 4, 5 | Paramètres de l'EPP llm-d | Hors périmètre (v2). |
| 7 | Nature de l'OSL P50 de 4K et de l'ISL P99 de 666K | Non vérifiable dans le code : l'app affiche un avertissement. |

## Correction importante : comptabilité du budget en speculative decoding (§2.4, §3.1)

La base de connaissances indique que DSpark/DFlash en 0.33 comptent « ~1 token par séquence » dans
`max_num_batched_tokens`. **C'est infirmé par le code source** :

- `vllm/v1/core/sched/scheduler.py` : pour une requête en cours, `num_new_tokens = num_tokens_with_spec −
  num_computed_tokens`. Les k tokens de draft à vérifier sont donc comptés dans `token_budget`, quelle que soit la méthode.
- `SpeculativeConfig.max_num_new_slots_for_drafting` : ce que #59468 supprime pour DSpark/DFlash, ce sont
  les **slots de drafting supplémentaires** (avant : K − 1 pour DSpark, K pour DFlash ; 0.33 : 0).
- La PR #59468 le confirme : DSpark K = 5 est « charged 6 tokens instead of 10 ».

Conséquence : `d = 1 + k` pour DSpark en 0.33 (et `d = 1 + k + (k − 1)` en 0.32). Au §3.1, les valeurs
« 51 / 42 / 36 » supposent `d = 1` ; avec `d = 4`, elles deviennent 49,6 / 41,3 / 35,4. Pour la config finale
(B = 8192, S = 512, a = 3), le ratio nécessaire est de **4,0**, et non d'environ 12 à 14.

#59468 ajoute aussi un plafond automatique : `max_num_active_seqs ≤ max_num_batched_tokens // num_lookahead_tokens`
(`vllm/config/vllm.py`), soit 8192 // 4 = 2048 dans le cas de référence, qui ne contraint donc pas les 512 séquences.

## Autres constats

- `max_cudagraph_capture_size` : sans réglage, vLLM retient `min(max_num_seqs × (k+1) × 2, 1024)` sur
  Blackwell datacenter (512 ailleurs), plafonné à `max_num_batched_tokens`. Avec spec decoding, la taille est
  arrondie à un multiple de k + 1 (`adjust_cudagraph_sizes_for_spec_decode`).
- Défauts en 0.33.0 : `gpu_memory_utilization` 0,92 ; `enable_prefix_caching` True ; `enable_chunked_prefill`
  True ; `long_prefill_token_threshold` 0 (désactivé) ; `draft_sample_method` `greedy` ;
  `rejection_sample_method` `standard`.
- FLy (`rejection_sample_method: "fly"`) et `draft_lm_head_quantization` n'existent qu'à partir de 0.33.0.
- `--enable-prefix-caching` et `--enable-chunked-prefill` sont des `BooleanOptionalAction` (forme `--no-…`).
- `vllm serve --help=all` est une syntaxe valide (`vllm/utils/argparse_utils.py`).
- `MistralLarge4ForConditionalGeneration` est absente du registre de modèles de 0.31.0, 0.32.0 et 0.33.0 ;
  la recette vLLM exige l'image `vllm/vllm-openai:mistral-large-4`.
- Non vérifié dans le code : le refus des flags inconnus par le frontend Rust (#62431) est repris des notes de version.
