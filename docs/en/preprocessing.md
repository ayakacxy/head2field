# Preprocessing profile

[English](preprocessing.md) · [简体中文](../zh/preprocessing.md)

Prepare a T1 image for Head2Field in three steps: align and resample the image,
build a CHARM head model, then assemble the model inputs. Install the
`preprocess` extra and SimNIBS 4.6.0 in the environment running the command.
Supply the T1 image and, for raw non-HCP data, an orientation template.

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

Use three new, distinct output directories; none may be inside another.
The command has no overwrite option. If a stage fails, results from completed
stages remain available, and existing user results are not deleted or replaced.

The Python entry point is `head2field.preprocessing.prepare_subject`.

## Stage 1: canonical T1

Select the profile that matches your input:

- `hcp_acpc`: for data with ACPC alignment and distortion correction already
  completed. It centers and resamples the head without estimating a rotation.
- `non_hcp_raw`: for a raw T1 image. Supply an orientation template for one
  six-degree-of-freedom rigid alignment; scaling and shear are rejected.

Both profiles write a float32, 256 x 256 x 256, 1 mm RAS T1 with affine
translation `[-128, -128, -128]`, a QC PNG, and a manifest with paths redacted.
To inspect the standalone canonicalization command, run:

```bash
python -m head2field.preprocessing canonicalize --help
```

## Stage 2: external CHARM

Head2Field calls `simnibs.segmentation.charm_main.run` from the installed
SimNIBS 4.6.0 package. A missing installation or a different version stops the
run. CHARM uses its own installed resources; this package does not bundle the
CHARM runtime or atlas, or substitute another segmentation method.

CHARM writes to a temporary directory beside the destination. Before moving
the completed result into place, Head2Field checks:

- the canonical T1 supplied from outside the head-model directory;
- CHARM's `T1.nii.gz` and `final_tissues.nii.gz` within that directory;
- exactly one `*.msh` head mesh at the directory root; and
- the installed SimNIBS version, which must be `4.6.0`.

`charm_runtime_v1.json` records these images, the head mesh, and the software
version. The assembly and FEM steps use it to confirm that their inputs belong
to the same head model.

Keep the complete CHARM output in its head-model directory. The next step
creates a separate, smaller subject directory for model inputs.

## Stage 3: prepared-subject assembly

Assembly reads the prepared T1, `final_tissues.nii.gz`, and CHARM head-model
directory. It first checks their association in `charm_runtime_v1.json`.

By default, electrode positions come from
`eeg_positions/EEG10-10_UI_Jurak_2007.csv`. This file must contain the standard
76 names and finite world-RAS coordinates. The saved cap uses the fixed order
with Oz first. To provide electrode files yourself, pass both
`--electrode-names-json` and `--electrode-coords-mm-npy`; the same checks apply.

Assembly then:

1. Checks the volume grid, affine, and tissue labels.
2. Copies the prepared T1 and tissue image and creates float32 scalar conductivity.
3. Saves GM/WM voxel queries in C-order, along with the ordered cap files.
4. Writes and validates the input manifest, checks that source files are
   unchanged, and moves the completed subject directory into place.

The subject directory contains copies of the prepared T1 and tissue image, not
the head mesh. For FEM planning and execution, pass the unique mesh from the
same CHARM directory. It is checked against `charm_runtime_v1.json` and the
input manifest's `head_model_provenance`.

The lower-level Python function is
`head2field.preprocessing.assemble_prepared_subject`.

## Small public inputs

For examples and software tests,
`head2field.preprocessing.synthetic.write_synthetic_preparation_inputs`
generates a deterministic 32-cube ellipsoid T1, an orientation image, and
synthetic spherical coordinates for the 76 standard names. These
non-anatomical inputs are intended for testing software operation, not for
measuring head-model or FEM accuracy.

## Testing

Synthetic tests cover profile selection, rigid-alignment calls, query order,
input preservation, and agreement between manifests and files. They also check
that failures preserve earlier results and that existing outputs are not
overwritten. To assess CHARM segmentation, run it on an actual subject and
record the SimNIBS version.
