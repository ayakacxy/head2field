# Contributing

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING_zh.md)

Bug reports, documentation fixes, and focused code changes are welcome.
Describe the problem, the resulting behavior, and the checks relevant to
your change.

## Report an issue

Include the software version, Python/PyTorch versions, operating system,
command or API call, and a small reproducible example. Synthetic inputs are
usually easiest to share. Remove participant identifiers, credentials, and
local storage paths from logs and attachments.

For a feature request, explain the research workflow it would support and
how it would affect the model's input and output formats.

## Development setup

Install an editable checkout from the repository root:

```bash
python -m pip install -e '.[dev]'
```

Work on preprocessing or FEM adapters requires a separate SimNIBS 4.6.0
installation. Real-data training requires CUDA-enabled PyTorch.
Unit tests use synthetic inputs; some integration tests need optional
dependencies.

Keep generated data, checkpoints, temporary files, caches, and build outputs
out of version control. The following commands place local checks in an
ignored task directory under `.work/`:

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

Choose checks that match the change. Inference and weight changes use
`tests/test_inference.py`, `tests/test_weights.py`, and
`tests/test_cli_smoke.py`; packaged-document changes use
`tests/test_packaging_resources.py`. Small changes can use the relevant test
files rather than the entire suite. Afterward, remove only files created for
your task.

Build release packages from a clean working copy with `python -m build`,
using `--outdir` to select a task-specific output directory.
The [release checklist](RELEASE_CHECKLIST.md) covers package verification.
Before changing model or training code, read the [code guide](en/code_guide.md)
for module responsibilities, tensor conventions, and data flow.

## Prepare a contribution

- Explain the observed behavior, intended behavior, and checks performed.
- Update the guides and examples when commands or input/output formats change.
- Use relative paths and fictional identifiers in examples and tests.
- Add regression tests for changes to numerical results, ordering, validation,
  or resume behavior. Report any optional tests that were skipped.
- Keep optional imports lazy so minimal inference does not load SimNIBS,
  SciPy, h5py, or the training pipeline.

<a id="scientific-compatibility"></a>

## Model compatibility

Existing weights and prepared data depend on parameter names and shapes,
tissue labels, conductivity scaling, world-RAS coordinates, electrode/query
order, the current convention, and output units. When changing any of these,
compare results on the same inputs and explain compatibility with existing
files. Incompatible changes require a new model or data-format version.

For optimizations, compare runtime, memory use, and numerical results on the
same inputs. Synthetic unit tests check software behavior; prediction accuracy
and generalization require appropriate evaluation data.

## Documentation language

Use English for public API names and CLI help, and Chinese for new source
comments and docstrings.

Technical guides live in `docs/en/` and `docs/zh/` with matching filenames.
Project guides in `docs/`, root READMEs, example guides, and weight guides have
`_zh.md` companions. Keep commands, scientific conventions, and training
settings aligned. Body links should stay in the current language, while the
language switch should open the same document in the other language.

Synchronize the documents shipped in `src/head2field/resources/` when updating
their source copies. The [documentation index](en/README.md) lists the guides.

## Attribution

The project author and maintainer is [ayakacxy](https://github.com/ayakacxy).
Codex (OpenAI) assists with development, documentation, and verification.
Record the actual commit author. For commits authored by ayakacxy with Codex
participation, add this trailer after a blank line:

```text
Co-authored-by: Codex <codex@openai.com>
```

## Data, dependencies, and licensing

Keep participant data, cohort manifests, complete training-resume states, and
generated research outputs out of source contributions. The bundled model
weights and checksums are package resources with documented architecture and
input requirements. See [data use](DATA_USE.md) and
[weights](../weights/README.md).

Record the source and license of copied or adapted code. Templates, atlases,
datasets, binaries, and other third-party assets need suitable distribution
terms. Contributions must be compatible with the
[GPL-3.0-only license](LICENSE_GUIDE.md).
