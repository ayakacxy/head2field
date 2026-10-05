"""Access configurations, schemas, documentation, and pretrained package weights."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any, Literal

ResourceKind = Literal["configs", "schemas", "docs", "examples", "release", "weights"]

CONFIG_FILES = (
    "fem_labels_v1.json",
    "head2field_v1.json",
    "preprocessing_v1.json",
    "training_smoke_v1.json",
    "training_v1.json",
)
SCHEMA_FILES = (
    "fem_labels_v1.schema.json",
    "prepared_subject_v1.schema.json",
    "training_dataset_v1.schema.json",
)
DOCUMENT_FILES = (
    "en/README.md",
    "en/code_guide.md",
    "en/data_contracts.md",
    "en/fem_label_generation.md",
    "en/getting_started.md",
    "en/inference.md",
    "en/input_contract.md",
    "en/model_architecture.md",
    "en/paper_reproduction.md",
    "en/preprocessing.md",
    "en/training.md",
    "zh/README.md",
    "zh/code_guide.md",
    "zh/data_contracts.md",
    "zh/fem_label_generation.md",
    "zh/getting_started.md",
    "zh/inference.md",
    "zh/input_contract.md",
    "zh/model_architecture.md",
    "zh/paper_reproduction.md",
    "zh/preprocessing.md",
    "zh/training.md",
)
EXAMPLE_FILES = ("README.md", "README_zh.md")
RELEASE_FILES = (
    "CHANGELOG.md",
    "CHANGELOG_zh.md",
    "CITATION.cff",
    "CITATION_GUIDE.md",
    "CITATION_GUIDE_zh.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING_zh.md",
    "DATA_USE.md",
    "DATA_USE_zh.md",
    "LICENSE",
    "LICENSE_GUIDE.md",
    "LICENSE_GUIDE_zh.md",
    "MODEL_CARD.md",
    "MODEL_CARD_zh.md",
    "README.md",
    "README_zh.md",
    "RELEASE_CHECKLIST.md",
    "RELEASE_CHECKLIST_zh.md",
    "SOURCE_PROVENANCE.md",
    "SOURCE_PROVENANCE_zh.md",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_NOTICES_zh.md",
)
WEIGHT_FILES = (
    "README.md",
    "README_zh.md",
    "head2field_v1_weights.pth",
)

_DECLARED_FILES: dict[ResourceKind, tuple[str, ...]] = {
    "configs": CONFIG_FILES,
    "schemas": SCHEMA_FILES,
    "docs": DOCUMENT_FILES,
    "examples": EXAMPLE_FILES,
    "release": RELEASE_FILES,
    "weights": WEIGHT_FILES,
}


def packaged_resource(kind: ResourceKind, name: str) -> Traversable:
    """Return a package resource validated against the allowlist."""

    normalized = PurePosixPath(name)
    if (
        normalized.is_absolute()
        or normalized.as_posix() != name
        or ".." in normalized.parts
        or "\\" in name
    ):
        raise ValueError(f"Resource name must be a normalized relative path: {name!r}")
    if name not in _DECLARED_FILES[kind]:
        raise ValueError(f"Undeclared {kind} resource: {name!r}")
    resource = files("head2field").joinpath("resources", kind, *normalized.parts)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Installation is missing resource: resources/{kind}/{name}"
        )
    return resource


def read_packaged_text(kind: ResourceKind, name: str) -> str:
    """Read an allowlisted package resource as UTF-8."""

    return packaged_resource(kind, name).read_text(encoding="utf-8")


def load_packaged_json(kind: Literal["configs", "schemas"], name: str) -> Any:
    """Parse a JSON configuration or schema shipped with the package."""

    return json.loads(read_packaged_text(kind, name))


@contextmanager
def materialized_resource(
    kind: ResourceKind,
    name: str,
) -> Iterator[Path]:
    """Expose a real resource path for external programs within a context."""

    with as_file(packaged_resource(kind, name)) as path:
        yield path


__all__ = [
    "CONFIG_FILES",
    "DOCUMENT_FILES",
    "EXAMPLE_FILES",
    "RELEASE_FILES",
    "SCHEMA_FILES",
    "WEIGHT_FILES",
    "ResourceKind",
    "load_packaged_json",
    "materialized_resource",
    "packaged_resource",
    "read_packaged_text",
]
