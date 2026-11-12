---
name: vllm-configurator-catalog
description: Ajoute ou met à jour un modèle (catalog/models/<id>.yaml) ou un GPU (catalog/gpus/<id>.yaml) dans le catalogue du projet vllm-configurator, en vérifiant chaque valeur à la source (config.json Hugging Face à révision épinglée, API HF, recettes vLLM, fiches constructeur) et en marquant verified/unverified sans jamais inventer de spécification. À utiliser dès que l'utilisateur veut supporter un nouveau modèle ou un nouveau GPU dans vllm-configurator, même formulé vaguement (« ajoute Llama 5 », « on passe sur H200 », « mets à jour la fiche Qwen avec la nouvelle révision », « le catalogue ne connaît pas ce modèle »), ou quand l'app signale un modèle ou un GPU inconnu.
---

# Ajouter un modèle ou un GPU au catalogue vllm-configurator

Le moteur de vllm-configurator calcule une configuration `vllm serve` à partir de ces fiches. Une valeur fausse
y devient une recommandation fausse en production, présentée avec l'autorité d'un calcul. D'où la règle
centrale : **chaque fait vient d'une source citée, et ce qui n'est pas confirmé est marqué `unverified`
ou laissé à `null`**. Une valeur absente fait demander une donnée à l'utilisateur, ce qui est bien
préférable à une valeur plausible mais inventée.

Les schémas font foi : `src/vllm_configurator/catalog/schema.py` (classes `Gpu` et `Model`). Les fiches
existantes servent d'exemples : `catalog/models/deepseek-v4.1-flash-1017.yaml` (architecture exotique,
KV non dérivable), `catalog/models/qwen3.8-flash-next.yaml` (KV calculé avec sa formule) et
`catalog/gpus/b300.yaml`. Le détail champ par champ est dans `references/fields.md` ; lis-le avant
d'écrire une fiche.

## Statuts : la règle de décision

Chaque bloc factuel porte `status`, `source` (URL ou description précise) et éventuellement `note`.

- `verified` : lu dans une source primaire. Exemples : `config.json` à une révision épinglée, l'API
  Hugging Face, le code source de vLLM à un tag, une recette officielle `recipes.vllm.ai` pour ses
  recommandations de service, une fiche technique du constructeur, des logs de démarrage réels.
- `unverified` : tout le reste. Cela couvre les sources secondaires (blogs, revendeurs), les valeurs
  calculées par toi, les déductions et les chiffres annoncés mais non confrontés à vLLM.

Un bloc n'est `verified` que si **toutes** ses valeurs le sont. S'il mélange les deux, mets
`unverified` et explique dans `note` ce qui est confirmé. Une déduction, même évidente (par exemple
« 1 tête KV, donc KV répliqué sur les rangs TP »), reste une déduction : écris-la dans `note`.

## Procédure : modèle

1. **Identifier le dépôt exact.** Les noms sont ambigus : Flash ou base, FP8, NVFP4 ou BF16, build daté.
   Liste les candidats (`https://huggingface.co/api/models?search=<nom>`). S'il en reste plusieurs
   plausibles, demande lequel à l'utilisateur au lieu de choisir. Note dans la fiche le checkpoint qui
   correspond à la demande.
2. **Collecter les faits** avec le script fourni, depuis la racine du repo :
   ```bash
   PYTHONUTF8=1 uv run python .claude/skills/vllm-configurator-catalog/scripts/hf_model_facts.py <org/repo> [--revision <sha>]
   ```
   Il affiche la révision à épingler, la taille du checkpoint, les champs utiles de `config.json`, les
   champs qui rendent le KV non standard, les indices de spec decoding natif (config et README), et le
   support de l'architecture dans chaque version de vLLM cataloguée. Une erreur HTTP 401 ou 404 signale
   un dépôt gated, privé ou mal nommé : demande à l'utilisateur, ne devine pas.
3. **Lire la recette vLLM** (`https://recipes.vllm.ai/<org/repo>`) pour la méthode et le `k` de spec
   decoding recommandés, le dtype du KV cache, la version minimale de vLLM, l'éventuelle image dédiée,
   les contraintes de taille de bloc et les avertissements. Recopie les recommandations de service dans
   `notes`, avec la mention « Recette vLLM : ».
4. **KV cache, le point le plus délicat.** Applique une formule standard seulement si le script ne
   signale aucun champ exotique (voir `references/fields.md`). Écris alors la formule développée dans
   `method` et mets `status: unverified`, car c'est un calcul. Si l'architecture est exotique
   (attention linéaire ou hybride, compression par couche, KV partagé, indexeurs, fenêtres glissantes),
   mets `bytes_per_token: null`, sauf valeur explicite d'une source que tu cites. Explique alors dans
   `method` pourquoi elle n'est pas dérivable. L'app exigera la valeur « GPU KV cache size » des logs,
   et c'est le comportement voulu.
5. **Taille de bloc.** Ne remplis `forced_by_backend` qu'à partir d'un log réel (« Setting kv cache
   block size to N ») ou d'une consigne explicite de la recette pour NVIDIA. Sinon, mets `null`.
6. **Support vLLM.** Si le script répond NON pour toutes les versions cataloguées, renseigne
   `serving.requires_custom_image` et `min_vllm_version` d'après la recette, et explique-le dans
   `serving.note`. Vérifie que `speculative.native_method` figure dans `speculative_methods` de
   `catalog/vllm/extracted/<version>.yaml` ; sinon, signale-le à l'utilisateur.
7. **Écrire** `catalog/models/<id>.yaml`. Le nom du fichier doit être égal à `id`. Utilise un `id` en
   minuscules et tirets, qui reprend le nom du dépôt, et indique la date du jour dans `verified_on`.

## Procédure : GPU

1. Cherche la fiche technique du constructeur : capacité HBM par GPU, compute capability (pour NVIDIA,
   `https://developer.nvidia.com/cuda-gpus`), bandes passantes. Les pages des systèmes (HGX, DGX)
   donnent souvent un total : divise-le, mais marque `unverified` si la valeur par GPU n'est pas
   écrite explicitement. Explique le recoupement dans `note`.
2. `cudagraph_family` : `blackwell_dc` pour toute compute capability 10.x, sinon `other`. vLLM ne
   regarde que la famille (`is_device_capability_family(100)` dans `vllm/config/vllm.py`).
3. `memory.hbm_gb` est en Go décimaux (10⁹ octets), comme les constructeurs. Si une source donne des
   Gio ou une capacité « utilisable », cite-la telle quelle dans `note` et ne la mélange pas avec la
   valeur nominale.
4. Écris `catalog/gpus/<id>.yaml`.

## Valider

```bash
uv run python -c "from vllm_configurator.catalog import load_catalog; c = load_catalog(); print(c.validate() or 'OK')"
uv run pytest tests -q
```

La première commande valide les schémas (champ manquant, type faux, champ inconnu) et la cohérence
entre fichiers (`id` différent du nom de fichier, méthode de spec decoding inconnue de vLLM). Corrige
toute erreur avant de conclure. Si des tests cassent parce qu'une fiche existante a changé, montre-les
à l'utilisateur au lieu d'ajuster les valeurs attendues.

## Compte rendu à l'utilisateur

Termine par un court tableau : chaque bloc, son statut, sa source. Liste ensuite explicitement :
- les valeurs laissées à `null` et ce qu'il faudrait fournir (en général la ligne « GPU KV cache
  size » et la taille de bloc des logs de démarrage) ;
- les contradictions entre sources (par exemple une recette qui recommande un `k` différent de celui
  qu'impose le checkpoint) ;
- un éventuel manque de support dans les versions de vLLM cataloguées.
