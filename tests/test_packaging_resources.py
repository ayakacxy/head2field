"""Validate release metadata and wheel resource contracts."""

from __future__ import annotations

import json
import os
import re
import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

from head2field.resource_paths import (
    CONFIG_FILES,
    DOCUMENT_FILES,
    EXAMPLE_FILES,
    RELEASE_FILES,
    SCHEMA_FILES,
    load_packaged_json,
    materialized_resource,
    packaged_resource,
    read_packaged_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_RELEASE_FILES = ("CITATION.cff", "LICENSE", "README.md", "README_zh.md")
RELEASE_GUIDE_FILES = tuple(
    name for name in RELEASE_FILES if name not in ROOT_RELEASE_FILES
)


def _citation_version() -> str:
    """Read the software version from the reviewed CFF top-level version field."""

    text = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*["\']?([^"\'\s]+)["\']?\s*$', text, re.M)
    if match is None:
        raise AssertionError("CITATION.cff 缺少唯一可解析的顶层 version")
    return match.group(1)


@pytest.mark.parametrize(
    ("kind", "names", "source_directory"),
    (
        ("configs", CONFIG_FILES, PROJECT_ROOT / "configs"),
        ("schemas", SCHEMA_FILES, PROJECT_ROOT / "schemas"),
        ("docs", DOCUMENT_FILES, PROJECT_ROOT / "docs"),
        ("examples", EXAMPLE_FILES, PROJECT_ROOT / "examples"),
        ("release", ROOT_RELEASE_FILES, PROJECT_ROOT),
        ("release", RELEASE_GUIDE_FILES, PROJECT_ROOT / "docs"),
        ("weights", ("README.md", "README_zh.md"), PROJECT_ROOT / "weights"),
    ),
)
def test_packaged_resources_match_reviewable_source_files(
    kind: str,
    names: tuple[str, ...],
    source_directory: Path,
) -> None:
    """包资源必须与仓库中对应的权威文件逐字节一致。"""

    assert names
    for name in names:
        expected = (source_directory / name).read_text(encoding="utf-8")
        assert read_packaged_text(kind, name) == expected
        with materialized_resource(kind, name) as path:
            assert path.is_file()
            assert path.read_text(encoding="utf-8") == expected


@pytest.mark.parametrize("kind", ("configs", "schemas"))
def test_packaged_json_resources_parse(kind: str) -> None:
    """Every shipped JSON resource must be parseable."""

    names = CONFIG_FILES if kind == "configs" else SCHEMA_FILES
    for name in names:
        assert load_packaged_json(kind, name) == json.loads(
            (PROJECT_ROOT / kind / name).read_text(encoding="utf-8")
        )


@pytest.mark.parametrize(
    "name",
    ("../head2field_v1.json", "/head2field_v1.json", "nested/head2field_v1.json", "."),
)
def test_packaged_resource_rejects_non_leaf_names(name: str) -> None:
    """Resource accessors must reject directory traversal and non-leaf paths."""

    with pytest.raises(ValueError):
        packaged_resource("configs", name)


def test_packaged_resource_rejects_undeclared_file() -> None:
    """Resource accessors must not expose files outside the allowlist."""

    with pytest.raises(ValueError, match="Undeclared"):
        packaged_resource("docs", "private_notes.md")


@pytest.mark.parametrize(
    "name",
    ("en/../zh/README.md", "/en/README.md", "en//README.md", "en/./README.md"),
)
def test_packaged_document_rejects_unsafe_or_noncanonical_paths(name: str) -> None:
    """Language directories must not permit traversal or alternate path spellings."""

    with pytest.raises(ValueError, match="normalized relative path"):
        packaged_resource("docs", name)


def test_release_candidate_metadata_and_entry_points() -> None:
    """Metadata must record the candidate version, PEP 639 license, and stable CLIs."""

    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    project = metadata["project"]
    assert project["name"] == "head2field"
    assert metadata["build-system"]["requires"][0] == "setuptools>=77"
    assert project["version"] == "0.1.0rc1"
    assert project["license"] == "GPL-3.0-only"
    assert project["license-files"] == ["LICENSE", "docs/THIRD_PARTY_NOTICES.md"]
    assert "Development Status :: 2 - Pre-Alpha" not in project["classifiers"]
    extras = project["optional-dependencies"]
    assert {"preprocess", "fem", "training", "full", "dev"} <= set(extras)
    assert "scipy>=1.11" in extras["preprocess"]
    assert extras["fem"] == ["h5py>=3.10"]
    assert extras["training"] == []
    assert set(extras["full"]) <= set(extras["dev"])
    assert project["scripts"] == {
        "head2field-predict": "head2field.cli:main",
        "head2field-prepare": "head2field.preprocessing.cli:prepare_main",
        "head2field-fem": "head2field.fem_labels.cli:main",
        "head2field-train": "head2field.training.cli:train_main",
        "head2field-train-preflight": ("head2field.training.cli:preflight_main"),
        "head2field-prepare-cache": ("head2field.training.cli:prepare_cache_main"),
        "head2field-prepare-statistics": (
            "head2field.training.cli:prepare_statistics_main"
        ),
        "head2field-prepare-physics": ("head2field.training.cli:prepare_physics_main"),
        "head2field-make-synthetic-training": (
            "head2field.training.cli:prepare_synthetic_main"
        ),
    }


def test_release_version_is_consistent_with_citation_and_install() -> None:
    """Source metadata, CFF, and installed distributions must share the same RC version."""

    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    expected = project["version"]
    assert expected == "0.1.0rc1"
    assert _citation_version() == expected

    require_installed = os.environ.get("HEAD2FIELD_RELEASE_REQUIRE_INSTALLED") == "1"
    try:
        installed = importlib_metadata.version("head2field")
    except importlib_metadata.PackageNotFoundError:
        if require_installed:
            raise AssertionError(
                "要求安装验证，但环境中没有候选 distribution"
            ) from None
    else:
        assert installed == expected


def test_sdist_manifest_includes_only_the_allowlisted_pretrained_weights() -> None:
    """The sdist includes default weights while excluding other checkpoints and local outputs."""

    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required_lines = {
        "include CITATION.cff",
        "include README.md",
        "include README_zh.md",
        "include .markdownlint.json",
        "graft configs",
        "graft schemas",
        "graft docs",
        "graft examples",
        "graft tests",
        "prune weights",
        "global-exclude *.pth",
        "include src/head2field/resources/weights/head2field_v1_weights.pth",
    }
    assert required_lines <= set(manifest.splitlines())
    assert manifest.index("global-exclude *.pth") < manifest.index(
        "include src/head2field/resources/weights/head2field_v1_weights.pth"
    )
