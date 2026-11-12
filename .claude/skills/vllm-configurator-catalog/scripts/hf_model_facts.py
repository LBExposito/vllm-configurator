"""Collect the facts needed for a catalog/models/<id>.yaml entry from Hugging Face.

Usage (from the repo root):
    uv run python .claude/skills/vllm-configurator-catalog/scripts/hf_model_facts.py <org/repo> [--revision <sha>]

Prints a report: pinned revision, checkpoint size, the config.json fields the
engine uses, the fields that make the KV cache non-standard, spec decoding
hints, and vLLM support per catalogued version. Downloaded files are parsed as
JSON/text only, never executed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
API = "https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
FILE = "https://huggingface.co/{repo}/resolve/{rev}/{name}"

# Keys whose presence means the simple KV formulas do not apply as-is.
EXOTIC_KEYS = [
    "layer_types", "sliding_window", "compress_ratios", "kv_source_layer_ids", "kv_source_layers",
    "linear_attn_config", "full_attention_interval", "index_topk", "indexer_types", "indexer_budget",
    "engram_layer_ids", "hc_mult", "no_rope_layers", "kv_lora_rank", "q_lora_rank",
]
SPEC_KEYS = ["num_nextn_predict_layers", "mtp_num_hidden_layers", "mtp", "dspark_block_size"]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-configurator-catalog"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def text_config(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg.get("language_config") or cfg


def kv_hint(tc: dict) -> str:
    layers = tc.get("num_hidden_layers")
    if "kv_lora_rank" in tc:
        rope = tc.get("qk_rope_head_dim", 0) or 0
        return (f"MLA : {layers} couches × (kv_lora_rank {tc['kv_lora_rank']} + qk_rope_head_dim {rope}) "
                f"× octets/élément, répliqué sur chaque rang TP — valable seulement si toutes les couches "
                f"sont des couches MLA sans partage de KV")
    kvh, hd = tc.get("num_key_value_heads"), tc.get("head_dim")
    if hd is None and tc.get("hidden_size") and tc.get("num_attention_heads"):
        hd = tc["hidden_size"] // tc["num_attention_heads"]
    if kvh and hd:
        return (f"GQA/MHA : 2 × {layers} couches × {kvh} têtes KV × head_dim {hd} × octets/élément, "
                f"divisé par min(TP, {kvh}) par GPU — valable seulement si toutes les couches ont une attention complète")
    return "aucune formule standard applicable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    try:
        info = json.loads(get(API.format(repo=args.repo, rev=args.revision)))
    except urllib.error.HTTPError as err:
        print(f"ERREUR : {args.repo}@{args.revision} inaccessible (HTTP {err.code}). "
              "Dépôt privé, gated ou nom incorrect : ne pas deviner, demander à l'utilisateur.")
        return 1
    sha = info["sha"]
    cfg = json.loads(get(FILE.format(repo=args.repo, rev=sha, name="config.json")))
    tc = text_config(cfg)
    sizes = [s.get("size", 0) for s in info.get("siblings", []) if s["rfilename"].endswith(".safetensors")]

    print(f"# {args.repo}")
    print(f"hf_revision: {sha}")
    print(f"lastModified: {info.get('lastModified')}")
    print(f"config_url: https://huggingface.co/{args.repo}/blob/{sha}/config.json")
    print(f"checkpoint_bytes (somme des .safetensors): {sum(sizes) if sizes else 'inconnu'}")
    print(f"parameters (API HF): {(info.get('safetensors') or {}).get('total', 'inconnu')}")
    print(f"quantization_config: {json.dumps(cfg.get('quantization_config', tc.get('quantization_config')), ensure_ascii=False)[:400]}")
    print("\n## Champs utilisés par le moteur")
    for key in ["architectures", "model_type"]:
        print(f"{key}: {cfg.get(key)}")
    for key in ["model_type", "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
                "head_dim", "max_position_embeddings", "n_routed_experts", "num_experts", "num_experts_per_tok",
                "n_shared_experts", "torch_dtype", "dtype"]:
        if key in tc:
            print(f"text.{key}: {tc[key]}")

    print("\n## Champs qui rendent le KV non standard (à analyser)")
    found = False
    for key in EXOTIC_KEYS:
        if key in tc:
            found = True
            value = tc[key]
            if isinstance(value, list) and len(value) > 12:
                uniq = sorted({json.dumps(v) for v in value})[:10]
                value = f"liste[{len(value)}], valeurs distinctes : {', '.join(uniq)}"
            print(f"text.{key}: {value}")
    if not found:
        print("(aucun)")
    print(f"\nPiste de formule KV : {kv_hint(tc)}")
    if found:
        print("ATTENTION : architecture non standard. La formule ci-dessus ne s'applique PAS telle quelle ; "
              "laisser bytes_per_token à null sauf source explicite (recette vLLM, fiche du modèle).")

    print("\n## Indices de spec decoding natif")
    for key in SPEC_KEYS:
        if key in tc:
            print(f"text.{key}: {tc[key]}")
    try:
        readme = get(FILE.format(repo=args.repo, rev=sha, name="README.md")).decode("utf-8", "replace")
        for line in readme.splitlines():
            if re.search(r"speculative-config|speculative_config|vllm/vllm-openai:|vllm serve", line):
                print(f"README: {line.strip()[:200]}")
    except urllib.error.HTTPError:
        print("README.md absent")

    print("\n## Support de l'architecture par version vLLM cataloguée")
    arch = (cfg.get("architectures") or ["?"])[0]
    for path in sorted((ROOT / "catalog" / "vllm" / "extracted").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        print(f"vLLM {data['version']}: {'oui' if arch in data['architectures'] else 'NON'} ({arch})")
    print(f"\nRecette vLLM à consulter : https://recipes.vllm.ai/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
