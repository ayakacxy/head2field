# Release checklist

[English](RELEASE_CHECKLIST.md) · [简体中文](RELEASE_CHECKLIST_zh.md)

Use this checklist when preparing a software release.

## Upload inventory

The repository root is the directory containing `pyproject.toml`.
Include the following files, together with their packaged copies under `src/`:

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

The wheel contains the installable Python package, resources, and pretrained
Head2Field weights. The source archive includes the same bundled weights.
Keep the weight notices, checksums, licenses, and data-use statements
consistent across both distributions.

`.gitignore` and the source manifest exclude `.work/`, caches, build outputs,
custom checkpoints, and generated research data. The bundled pretrained
weight is the intended exception.

## Version and documentation

- [ ] Match the software version in `pyproject.toml`, `CITATION.cff`, package
  metadata, and release notes.
- [ ] Update `docs/CHANGELOG.md` with user-visible changes and compatibility notes.
- [ ] Align English and Chinese documents, matching filenames under `docs/en/`
  and `docs/zh/`, same-language body links, and same-document language switches.
- [ ] Check installation commands, CLI examples, and relative links against
  the release source.
- [ ] Synchronize packaged configs, schemas, and documents with their source copies.
- [ ] Use verified contributor, repository, paper, and DOI metadata.
  Include available information without inventing identifiers.

## Software verification

- [ ] Run the complete test suite with the required SimNIBS and CUDA dependencies.
  Resolve failures and investigate skipped tests before release.
- [ ] Check inference, preprocessing, FEM data validation, training, resume,
  CLI entry points, and packaged resources.
- [ ] Run Markdown lint with the repository configuration and check local links.
  Keep only the two READMEs at the repository root.
- [ ] Record software versions, GPU, test results, and package checksums.
  Use evaluation data to measure accuracy; synthetic tests check software behavior.

## Build and artifact contents

- [ ] Build the reviewed source in a task directory under `.work/`.
  Set temporary, cache, and output paths to that directory.
- [ ] Build the source archive first, then build a wheel from its extracted
  contents to check that the source distribution is complete.
- [ ] Inspect both archives for code, resources, documents, licenses, citation
  metadata, and the single bundled pretrained weight.
- [ ] Exclude local caches, test outputs, raw participant data, generated research
  results, and custom checkpoints. Compare source/resource copies byte for byte.
- [ ] Install the wheel in an isolated task directory, run resource and metadata
  tests against the installation, and check all nine CLI help commands.
- [ ] Generate `SHA256SUMS` for the wheel and source archive.

## Model weights and data-use notices

- [ ] Include licenses, data-use notices, versions, model descriptions,
  and checksums with the weights.
- [ ] Check that the file contains only the expected model tensors, with no
  optimizer, scheduler, scaler, step counts, metrics, identifiers, or unused
  tensor-storage payloads.
- [ ] Include dataset acknowledgments and method references. For HCP data,
  follow the official guidance linked in [data use](DATA_USE.md).
- [ ] State which notices cover the code, weights, and upstream datasets.

## Distribution verification

- [ ] Commit reviewed changes with contributor attribution, then create a version
  tag pointing to that source commit.
- [ ] Publish the tag, release notes, wheel, source archive, and `SHA256SUMS`.
  Match the tag to the software version and disable the prerelease flag for
  a formal release.
- [ ] Retain the repository visibility chosen by the owner. Check repository,
  release, citation, and download links with the appropriate access permissions.
- [ ] Read back the release name, tag, draft/prerelease state, commit, and assets.
  Download the files and compare their SHA-256 values with the recorded checksums.
- [ ] Update both README release badges and links to the published version.
