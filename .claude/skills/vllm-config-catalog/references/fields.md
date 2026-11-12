# Fiches du catalogue : guide champ par champ

Le schéma `src/vllm_config/catalog/schema.py` fait foi ; ce guide explique d'où tirer chaque valeur.

## Fiche modèle (`catalog/models/<id>.yaml`)

| Champ | Source | Remarques |
|---|---|---|
| `hf_repo`, `hf_revision` | Script (`sha` de l'API HF) | Toujours épingler le SHA de 40 caractères, jamais `main`. |
| `architecture.hf_architecture` | `config.json` → `architectures[0]` | Sert à vérifier le support dans `catalog/vllm/extracted/*.yaml`. |
| `architecture.family` | `dense` ou `moe` | `moe` exige le bloc `moe`. |
| `architecture.attention` | Description libre, en snake_case | Ex. `gqa`, `mla`, `sparse_dsa`, `sliding_window`, `linear_kda`, `kv_sharing`. |
| `num_hidden_layers` … `max_position_embeddings` | `config.json` (bloc `text_config` des modèles multimodaux) | Recopier tel quel. Une valeur sentinelle (`head_dim: 0`) devient `null`, avec la raison dans `note`. |
| `moe.*` | `n_routed_experts` ou `num_experts`, `num_experts_per_tok`, `n_shared_experts` | Un nombre d'experts partagés déduit (par exemple de `shared_expert_intermediate_size`) va dans `note`. |
| `architecture.raw` | Champs bruts utiles mais non interprétés | Résumer les longues listes (valeurs distinctes, comptes). |
| `weights.checkpoint_bytes` | Script (somme des `.safetensors`) | Préciser la quantization lue dans `quantization_config`. |
| `kv_cache.*` | Voir ci-dessous | Presque toujours `unverified`. |
| `block_size.forced_by_backend` | Logs de démarrage ou recette (NVIDIA) | `null` par défaut. |
| `speculative.*` | `config.json` (champs `mtp*`, `dspark_*`, `num_nextn_predict_layers`), README, recette | `fixed_k: true` si le checkpoint impose la profondeur du draft. |
| `serving.*` | Recette vLLM | `requires_custom_image` si aucune release ne prend le modèle en charge. |

### Formules KV standard (uniquement sans champ exotique)

- **MHA / GQA** : `2 × num_hidden_layers × num_key_value_heads × head_dim × octets(dtype)` par token.
  Par GPU, diviser par `min(TP, num_key_value_heads)` : `replicated_across_tp: false`.
- **MLA (style DeepSeek V2/V3)** : `num_hidden_layers × (kv_lora_rank + qk_rope_head_dim) × octets(dtype)`
  par token, répliqué sur chaque rang TP (`replicated_across_tp: true`). La disposition `fp8_ds_mla` de
  vLLM ajoute des facteurs d'échelle : la formule est donc une estimation basse.
- **État par séquence** (attention linéaire, Mamba, Gated DeltaNet) : à mettre dans
  `per_sequence_state_bytes` seulement si les dimensions et le dtype de l'état figurent explicitement
  dans `config.json`. Détailler le calcul dans `method`.

Champs exotiques qui invalident ces formules : `layer_types` mixtes, `compress_ratios`,
`kv_source_layer_ids` / `kv_source_layers` (KV partagé), `linear_attn_config`,
`full_attention_interval`, `sliding_window`, indexeurs (`index_topk`, `indexer_types`), `engram_*`. En
leur présence, `bytes_per_token: null`, sauf chiffre explicite d'une source citée.

## Fiche GPU (`catalog/gpus/<id>.yaml`)

| Champ | Source | Remarques |
|---|---|---|
| `vendor`, `generation` | Constructeur | `generation` en snake_case (`hopper`, `blackwell`, `blackwell_ultra`). |
| `cudagraph_family` | Dérivé de la compute capability | `blackwell_dc` pour 10.x, sinon `other`. |
| `compute_capability` | `https://developer.nvidia.com/cuda-gpus` | `unverified` si la valeur ne vient pas de cette page. |
| `memory.hbm_gb` | Fiche technique, en Go décimaux | Les sources secondaires et les totaux divisés restent `unverified`. |
| `bandwidth.*` | Fiche technique | Champs facultatifs. Un bloc qui mélange des valeurs confirmées et non confirmées est `unverified`, avec une `note`. |
