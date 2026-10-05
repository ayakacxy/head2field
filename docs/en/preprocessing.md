# Preprocessing profile

[English](preprocessing.md) · [简体中文](../zh/preprocessing.md)

The preprocessing workflow converts a caller-supplied T1 image into a validated
`prepared_subject_v1` directory through canonicalization, external CHARM head
modeling, and prepared-subject assembly. Install the `preprocess` extra and
SimNIBS 4.6.0 in the active environment. Templates and participant inputs are
provided by the user.

## End-to-end command

```bash
head2field-prepare \
  --input-t1 raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template orientation_template.nii.gz \
  --canonical-output-dir canonical_stage \
  --head-model-output-dir charm_head_model \
  --prepared-output-dir prepared_subject
```

The three output paths must be new, distinct, and non-nested. There is no
overwrite option. A completed earlier stage is retained if a later stage fails;
the tool never deletes or rewrites a caller-owned result.

The same workflow is available as
`head2field.preprocessing.prepare_subject`.

## Stage 1: canonical T1

Two explicit profiles are supported:

- `hcp_acpc` assumes ACPC alignment and distortion correction are already
  complete. It estimates no rotation and only centers/resamples the head onto
  the fixed grid.
- `non_hcp_raw` requires a caller-provided orientation template and performs
  exactly one six-degree-of-freedom rigid estimate. Scaling and shear are
  rejected.

Both profiles produce a float32, 256 x 256 x 256, 1 mm RAS T1 with affine
translation `[-128, -128, -128]`, a QC PNG, and a path-redacted manifest.
The lower-level command is:

```bash
python -m head2field.preprocessing canonicalize --help
```

## Stage 2: external CHARM

The default adapter imports SimNIBS 4.6.0 from the user's environment and calls
`simnibs.segmentation.charm_main.run`. A version mismatch or missing runtime is
an error; there is no fallback segmentation. CHARM uses resources from that
external installation. No runtime or atlas is copied into this package.

The head model is generated in sibling staging and promoted only after
all of the following identity inputs exist:

- the caller's canonical T1 outside the head-model directory;
- CHARM's own `T1.nii.gz` and `final_tissues.nii.gz` inside that directory;
- exactly one root-level `*.msh` head mesh; and
- the exact external SimNIBS version, which must be `4.6.0`.

Runtime checks compare parameters, groups, relative paths, file sizes, coordinate order, shapes, dtypes and finite values directly. Resume compares the full configuration and dataset manifest. Regenerate statistics for the current 50/100 mm regions and training split.

The resulting head-model directory is a durable external runtime output,
separate from the compact prepared-subject directory.

## Stage 3: prepared-subject assembly

Assembly accepts the canonical T1, `final_tissues.nii.gz`, and the current CHARM
head-model directory. It first validates `charm_runtime_v1.json` and requires
the supplied T1 and tissues to match that exact CHARM runtime identity. By
default it then reads
`eeg_positions/EEG10-10_UI_Jurak_2007.csv`, requires exactly the fixed 76-name
set, and reorders its finite world-RAS coordinates to the frozen list with Oz
first. Callers therefore do not have to pre-create name or coordinate files.
The paired `--electrode-names-json` and `--electrode-coords-mm-npy` options are
retained only as an explicit, equally strict adapter override. Assembly then:

The prepared directory copies only its prepared T1 and final-tissues inputs; it
does not copy or silently select a mesh. The mesh stays in the external CHARM
head-model directory. Downstream FEM planning and execution must provide that
same unique root mesh and revalidate it against both `charm_runtime_v1.json` and
the prepared manifest's `head_model_provenance`.

The lower-level API is
`head2field.preprocessing.assemble_prepared_subject`.

## Small public inputs

Examples and tests can call
`head2field.preprocessing.synthetic.write_synthetic_preparation_inputs`.
It writes a deterministic 32-cube ellipsoid T1, a second orientation image, and
synthetic spherical coordinates for the fixed 76 names. These are non-anatomical
inputs and are not evidence that CHARM or FEM output is scientifically valid.

## Validation boundary

Synthetic tests cover profile routing, one-call 6-DOF behavior, C-order query
assembly, source immutability, strict runtime-manifest fields, cross-manifest
identity binding, and failure/no-overwrite transactions. Real
SimNIBS/CHARM execution remains an external-runtime integration check and must
be recorded with the exact installed version for a release validation run.
