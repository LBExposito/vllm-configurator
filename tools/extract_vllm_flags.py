"""Extract the CLI flags and JSON config keys of a vLLM release from its source.

Usage:
    python tools/extract_vllm_flags.py v0.33.0 [--cache-dir .cache/vllm-src]

Downloads a few source files at the given git tag (read-only, parsed with
regexes, never imported or executed) and writes
`catalog/vllm/extracted/<version>.yaml`. The rules engine refuses to emit a
flag or a JSON key that is not listed in this file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import urllib.request
from pathlib import Path

import yaml

RAW = "https://raw.githubusercontent.com/vllm-project/vllm/{tag}/{path}"
ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "arg_utils": "vllm/engine/arg_utils.py",
    "speculative": "vllm/config/speculative.py",
    "compilation": "vllm/config/compilation.py",
    "registry": "vllm/model_executor/models/registry.py",
}


def fetch(tag: str, path: str, cache_dir: Path) -> str:
    target = cache_dir / tag / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RAW.format(tag=tag, path=path), timeout=60) as resp:
            target.write_bytes(resp.read())
    return target.read_text(encoding="utf-8")


def cli_flags(arg_utils: str) -> list[str]:
    # Every literal "--xxx" passed to an add_argument(...) call.
    flags: set[str] = set()
    for call in re.finditer(r"add_argument\((.*?)\*\*|add_argument\((.*?)\)", arg_utils, re.S):
        body = call.group(1) or call.group(2) or ""
        flags.update(re.findall(r"\"(--[a-z0-9][a-z0-9-]*)\"", body))
    return sorted(flags)


def class_fields(source: str, class_name: str) -> list[str]:
    match = re.search(rf"^class {class_name}\b.*?:\n(.*?)(?=^class |\Z)", source, re.S | re.M)
    if not match:
        return []
    return sorted(set(re.findall(r"^    ([a-z_][a-z0-9_]*): ", match.group(1), re.M)))


def literal_values(source: str, name: str) -> list[str]:
    match = re.search(rf"^{name} = Literal\[(.*?)\]", source, re.S | re.M)
    return sorted(set(re.findall(r"\"([a-z0-9_]+)\"", match.group(1)))) if match else []


def architectures(registry: str) -> list[str]:
    return sorted(set(re.findall(r"^\s+\"([A-Za-z0-9]+(?:ForCausalLM|ForConditionalGeneration))\"\s*:", registry, re.M)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="git tag, e.g. v0.33.0")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "vllm-src")
    args = parser.parse_args()

    src = {key: fetch(args.tag, path, args.cache_dir) for key, path in FILES.items()}
    spec_methods = set(literal_values(src["speculative"], "SpeculativeMethod"))
    for alias in ("MTPModelTypes", "EagleModelTypes", "DFlashModelTypes", "DSparkModelTypes", "NgramGPUTypes"):
        spec_methods.update(literal_values(src["speculative"], alias))

    data = {
        "version": args.tag.removeprefix("v"),
        "source": {
            "repository": "https://github.com/vllm-project/vllm",
            "tag": args.tag,
            "files": list(FILES.values()),
            "extracted_on": dt.date.today().isoformat(),
            "tool": "tools/extract_vllm_flags.py",
        },
        "cli_flags": cli_flags(src["arg_utils"]),
        "speculative_config_keys": class_fields(src["speculative"], "SpeculativeConfig"),
        "speculative_methods": sorted(spec_methods),
        "rejection_sample_methods": literal_values(src["speculative"], "RejectionSampleMethod"),
        "draft_sample_methods": literal_values(src["speculative"], "DraftSampleMethod"),
        "compilation_config_keys": class_fields(src["compilation"], "CompilationConfig"),
        "architectures": architectures(src["registry"]),
    }
    out = ROOT / "catalog" / "vllm" / "extracted" / f"{data['version']}.yaml"
    out.write_text(
        "# Fichier généré par tools/extract_vllm_flags.py : ne pas éditer à la main.\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    print(f"{out.relative_to(ROOT)}: {len(data['cli_flags'])} flags, "
          f"{len(data['speculative_config_keys'])} speculative keys, "
          f"{len(data['compilation_config_keys'])} compilation keys, "
          f"{len(data['architectures'])} architectures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
