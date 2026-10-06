# Release checklist

[English](RELEASE_CHECKLIST.md) · [简体中文](RELEASE_CHECKLIST_zh.md)

Maintainers can use this checklist to prepare and verify a software release.

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

The wheel is an installable release asset containing Python code, resources,
and the pretrained Head2Field v1 weights. The source archive contains the same
allowlisted weights and can be published alongside it. Keep weight notices,
checksums, licenses, and data-use statements aligned across both distributions.

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

- [ ] Run the complete test suite in an environment with the required SimNIBS
  and CUDA dependencies; investigate failures and skipped tests before release.
- [ ] Check inference, preprocessing, FEM contracts, training, resume, CLI
  entry points, and package resource consistency.
- [ ] Run Markdown lint with the repository configuration and validate local
  links. Keep only the two READMEs at the root.
- [ ] Record the actual software versions, GPU, test results, and artifact
  checksums. Synthetic tests verify software behavior; model accuracy needs evaluation data.

## Build and artifact contents

- [ ] Build from the reviewed source in a task directory under `.work/`; direct
  temporary files, caches, and outputs into that directory.
- [ ] Build the source archive first, then build the wheel from the extracted
  archive to verify that the source distribution is complete.
- [ ] Inspect both archives for the intended code, resources, documents,
  licenses, citation metadata, and the single allowlisted pretrained weight.
- [ ] Exclude local caches, test outputs, raw subject data, generated research
  results, and custom checkpoints; verify source/resource byte equality.
- [ ] Install the wheel into an isolated task directory, run resource and
  metadata tests against that installation, and check all nine CLI help commands.
- [ ] Generate `SHA256SUMS` for the wheel and source archive.

## Model weights and data-use notices

- [ ] Include applicable licenses, data-use notices, versions, model contracts,
  and checksums with the bundled weights.
- [ ] Confirm that weights contain only the expected model tensors, with no
  optimizer, scheduler, scaler, step counts, metrics, identifiers, or unused
  tensor-storage payloads.
- [ ] Provide dataset acknowledgments and relevant method references. For HCP,
  follow the official acknowledgment guidance linked in [data use](DATA_USE.md).
- [ ] Distinguish the code license, weight notices, and upstream dataset terms
  in release documentation.

## Distribution verification

- [ ] Commit reviewed changes with the appropriate contributor attribution;
  create a version tag pointing to the verified source commit.
- [ ] Publish the tag, release notes, wheel, source archive, and `SHA256SUMS`.
  The tag must match the software version; formal releases disable the prerelease flag.
- [ ] Preserve the repository visibility selected by the owner. Check repository,
  release, citation, and download links with the appropriate access permissions.
- [ ] Read back the release name, tag, draft/prerelease state, commit, and assets.
  Download the assets and compare their SHA-256 values with the recorded checksums.
- [ ] Update both README release badges to link to the published version.
