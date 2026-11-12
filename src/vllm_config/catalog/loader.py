"""Load and cross-validate the YAML catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .schema import ExtractedVersion, Gpu, Model, VllmVersion

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "catalog"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


@dataclass(frozen=True)
class Catalog:
    gpus: dict[str, Gpu]
    models: dict[str, Model]
    vllm_versions: dict[str, VllmVersion]
    extracted: dict[str, ExtractedVersion]

    def sorted_versions(self) -> list[str]:
        return sorted(self.vllm_versions, key=version_key)

    def validate(self) -> list[str]:
        """Cross-file checks. Returns a list of errors (empty when consistent)."""
        errors: list[str] = []
        for version, vc in self.vllm_versions.items():
            ex = self.extracted.get(version)
            if ex is None:
                errors.append(f"vllm {version}: extracted/{version}.yaml manquant (lancer tools/extract_vllm_flags.py)")
                continue
            for name, spec in vc.flags.items():
                if spec.cli not in ex.cli_flags:
                    errors.append(f"vllm {version}: flag {spec.cli} ({name}) absent du code source au tag {vc.source_tag}")
        for model in self.models.values():
            method = model.speculative.native_method
            for version, ex in self.extracted.items():
                if method and method not in ex.speculative_methods and version_key(version) >= version_key(
                    model.serving.min_vllm_version or "0.0.0"
                ):
                    errors.append(f"modèle {model.id}: méthode de spec decoding {method!r} inconnue de vLLM {version}")
        for gpu_id, gpu in self.gpus.items():
            if gpu.id != gpu_id:
                errors.append(f"gpu {gpu_id}: le nom du fichier doit être <id>.yaml (id={gpu.id})")
        for model_id, model in self.models.items():
            if model.id != model_id:
                errors.append(f"modèle {model_id}: le nom du fichier doit être <id>.yaml (id={model.id})")
        return errors


def load_catalog(root: Path = DEFAULT_ROOT) -> Catalog:
    gpus = {p.stem: Gpu(**_load_yaml(p)) for p in sorted((root / "gpus").glob("*.yaml"))}
    models = {p.stem: Model(**_load_yaml(p)) for p in sorted((root / "models").glob("*.yaml"))}
    versions = {p.stem: VllmVersion(**_load_yaml(p)) for p in sorted((root / "vllm").glob("*.yaml"))}
    extracted = {p.stem: ExtractedVersion(**_load_yaml(p)) for p in sorted((root / "vllm" / "extracted").glob("*.yaml"))}
    return Catalog(gpus=gpus, models=models, vllm_versions=versions, extracted=extracted)


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    return load_catalog()
