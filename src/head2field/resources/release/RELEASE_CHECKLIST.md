# Release checklist

[English](RELEASE_CHECKLIST.md) · [简体中文](RELEASE_CHECKLIST_zh.md)

Use this checklist for each software release. It is a repeatable maintainer
workflow, not a historical record of completed checks.

## Upload inventory

Use the directory containing `pyproject.toml` as the repository root. Commit the following files
and directories, including the packaged resource copies under `src/`:

| Contents | Repository paths |
| --- | --- |
| Python package and packaged resources | `src/` |
| Default configurations and schemas | `configs/`, `schemas/` |
| Project guides, technical documentation, and overview figure (PNG and PDF) | `docs/` |
| Synthetic example generators and tests | `examples/`, `tests/` |
| English and Chinese project overviews | `README.md`, `README_zh.md` |
| Citation and license | `CITATION.cff`, `LICENSE` |
| Installation and source packaging | `pyproject.toml`, `MANIFEST.in`, `.gitignore` |
| Bundled pretrained model and metadata | `src/head2field/resources/weights/` |

Runtime checks compare parameters, groups, relative paths, file sizes, coordinate order, shapes, dtypes and finite values directly. Resume compares the full configuration and dataset manifest. Regenerate statistics for the current 50/100 mm regions and training split.

The `.gitignore` and source manifest exclude `.work/`, caches, build outputs,
custom checkpoints and generated research data from the code distribution.
The verified pretrained v1 weight file is explicitly included.

## Version and documentation

- [ ] Keep the software version consistent across `pyproject.toml`,
  `CITATION.cff`, package metadata, and release notes.
- [ ] Review `docs/CHANGELOG.md` and document user-visible compatibility changes.
- [ ] Keep English and Chinese documents aligned, including matching filenames
  under `docs/en/` and `docs/zh/`; verify
  same-language body links and same-document language switches.
- [ ] Verify installation commands, CLI examples, and relative documentation
  links against the release source.
- [ ] Synchronize package resources with top-level configs, schemas, and docs.
- [ ] Use verified contributor, repository, paper, and DOI metadata; include
  those fields when available rather than inventing identifiers.

## Software verification

## Build and artifact contents

## Model weights and data-use notices

## Distribution verification
