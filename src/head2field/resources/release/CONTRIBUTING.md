# Contributing

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING_zh.md)

Contributions to Head2Field are welcome. Please keep changes
focused, describe the problem they solve, and include verification appropriate
to their scope.

## Report an issue

For a bug report, include the software version, Python/PyTorch versions,
operating system, relevant command or API call, and a small reproducible example.
Use synthetic inputs when possible. Remove participant identifiers, credentials,
and local storage paths from logs and attachments.

For a feature request, describe the intended research workflow and how the
proposed change interacts with the v1 input and output formats.

## Development setup

From the repository root of a source checkout:

```bash
python -m pip install -e '.[dev]'
```

Install SimNIBS 4.6.0 separately when working on its preprocessing or FEM
adapters. Real-data training requires CUDA-enabled PyTorch. Unit tests use
synthetic inputs; integration tests may require optional dependencies.

Keep generated data, checkpoints, temporary files, caches, and build outputs
out of version control. Use a task directory under `.work/` for local checks:

```bash
mkdir -p .work/dev/tmp .work/dev/cache .work/dev/pycache
export TMPDIR="$PWD/.work/dev/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$PWD/.work/dev/cache"
export PYTHONPYCACHEPREFIX="$PWD/.work/dev/pycache"
export PIP_CACHE_DIR="$PWD/.work/dev/cache/pip"
export RUFF_CACHE_DIR="$PWD/.work/dev/cache/ruff"
export TORCH_HOME="$PWD/.work/dev/cache/torch"

python -m pytest \
  --basetemp .work/dev/pytest \
  -o cache_dir=.work/dev/cache/pytest
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m compileall -q src tests examples
```

Build release packages from a clean working copy with `python -m build`; direct
its output to a task-specific directory with `--outdir`. The
[release checklist](RELEASE_CHECKLIST.md) describes packaging verification.

For a focused change, select the relevant test files instead of rerunning the
whole suite. For example, inference/weight changes use
`tests/test_inference.py`, `tests/test_weights.py`, and `tests/test_cli_smoke.py`;
packaged-document changes use `tests/test_packaging_resources.py`.
After checks finish, remove only the task files and directories you created.

Read the [code guide](en/code_guide.md) for the package map, tensor conventions,
and data flow before changing model or training code.

## Prepare a contribution

- Explain the observed behavior, intended behavior, and relevant verification.
- Update documentation and examples when commands or input/output formats change.
- Use relative paths and fictional identifiers in examples and tests.
- Include regression tests for changes that affect numerical results, ordering,
  validation, or resume behavior. State optional tests that were skipped.
- Preserve lazy optional imports so minimal inference does not load SimNIBS,
  SciPy, h5py, or the training pipeline.

<a id="scientific-compatibility"></a>

## Model compatibility

The v1 model requires consistent parameter names and shapes, tissue labels,
conductivity scaling, world-RAS coordinates, electrode/query order, current
convention, and output units. When changing these, compare numerical results
on the same inputs and explain compatibility with existing weights and data.
Incompatible changes need a new model or data-format version and documentation.

For optimization work, compare the same inputs and report runtime, memory use,
and numerical differences. Synthetic unit tests establish software behavior;
scientific accuracy and generalization need appropriate evaluation data.

## Documentation language

Public API names and CLI help use English. New source comments and docstrings
use Chinese.
Maintain technical guides in `docs/en/` and `docs/zh/` with matching filenames.
Project guides in `docs/`, root READMEs, example documents, and weight documents
retain their `_zh.md` companions.
Keep commands, scientific conventions, and training settings aligned.
Body links stay in the current language; the language
switch opens the same document in the other language. Synchronize shipped
documents with their copies in `src/head2field/resources/`. The
[documentation index](en/README.md) lists all guides.

## Attribution

The project author and maintainer is [ayakacxy](https://github.com/ayakacxy).
Codex (OpenAI) contributes development, documentation, and verification
assistance. Record the actual author of each commit. For commits authored by
ayakacxy with Codex participation, add the following trailer to the commit
message after a blank line:

```text
Co-authored-by: Codex <codex@openai.com>
```

## Data, dependencies, and licensing

Do not include participant data, cohort manifests, complete resume states, or
generated research outputs in source changes. The bundled v1 weights and their
checksums are package resources with documented architecture and input requirements. See
[data use](DATA_USE.md) and [weight documentation](../weights/README.md).

Document the origin and license of copied or adapted code. Add no templates,
atlases, datasets, binaries, or other third-party assets without appropriate
distribution terms. Contributions must be compatible with the project's
[GPL-3.0-only license](LICENSE_GUIDE.md).
