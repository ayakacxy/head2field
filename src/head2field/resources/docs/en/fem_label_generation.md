# FEM label generation

[English](fem_label_generation.md) · [简体中文](../zh/fem_label_generation.md)

Generate training labels from a prepared subject and its CHARM head mesh using
SimNIBS 4.6.0. The workflow computes 75 source-to-Oz fields with the standard
76-electrode cap, circular electrodes of diameter 10 mm and thickness 4 mm,
and the full isotropic tissue conductivity table. Choose new output directories.

Use `plan` to print the simulation settings without writing files, `generate`
to run the solver and export labels, and `validate` to check the result:

```bash
head2field-fem plan --prepared-subject-dir prepared_subject --head-mesh head_model/subject.msh

head2field-fem generate   --prepared-subject-dir prepared_subject   --head-mesh head_model/subject.msh   --leadfield-output-dir leadfield_run   --labels-output-dir fem_labels

head2field-fem validate   --prepared-subject-dir prepared_subject   --fem-labels-dir fem_labels
```

If a validated solver run already exists, use `export` to interpolate its
fields at the GM/WM query points. Run `head2field-fem export --help` for the
required HDF5, solver-log, runtime-manifest, and destination arguments.
The interpolation is linear. SimNIBS fields for 1 A are multiplied by 0.001 to
obtain the source `+1 mA`, Oz `-1 mA` convention. Labels contain float32
world-RAS vectors in `V/m`.

Keep the solver log, HDF5 output, and runtime MAT file. Validation compares
the conductivity settings with the values read back from the MAT file.
The label manifest records source order, relative paths, file sizes, and
`[Q,3]` array shapes. Arrays must be float32 and contain only finite values.
Exports also save `query_ijk.npy`, whose ordered coordinates are compared
directly with the prepared subject.

The validator checks currents, Oz reference, field type, units, tissues,
source order, coordinates, conductivity, arrays, and paths. It uses the same
checks for every run; optional digests in older manifests are not acceptance
criteria. Results are moved from a temporary sibling directory into the new
destination only when complete. Existing results are never overwritten.

See [data formats and checks](data_contracts.md) and the
[FEM schema](../../schemas/fem_labels_v1.schema.json) for the stored fields.
