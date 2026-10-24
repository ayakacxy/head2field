<a id="data-contracts"></a>

# Data formats and validation

[English](data_contracts.md) · [简体中文](../zh/data_contracts.md)

Manifests describe prepared inputs and training data. Their JSON schemas list
the required fields; the loaders also inspect the referenced arrays for matching
shapes, coordinates, conductivity, electrode order, and units. Numerical arrays
containing NaN or Inf are rejected.

| Format | Purpose |
| --- | --- |
| `prepared_subject_v1` | Prepared inputs for inference, FEM, and training |
| `fem_labels_v1` | FEM labels for all 75 sources, used in training |
| `training_loss_statistics_v2` | Energy statistics for the 50/100 mm regions |
| `training_dataset_v1` | Training file paths and train/validation/test groups |

Use the [schemas](../../schemas/) for field definitions and the
[input-format guide](input_contract.md) for the directory layout. Write each
file path relative to its manifest. The loader checks that paths are safe,
files exist, and recorded sizes match.

Keep query rows and source order identical across predictions, FEM labels,
and training statistics:

- Query rows follow `gmwm_ijk`; sources follow the electrode list with Oz removed.
- FEM export saves `query_ijk.npy` and compares it with the prepared query rows.
- Energy-statistics NPZ files store query rows, the affine, and electrode
  coordinates, which are compared with the inputs when loaded.

Loading checks all conductivity-table values and their units. FEM checks also
cover current, vector components, HDF5 metadata, and the conductivity actually
used by the solver. To resume training, keep the saved configuration, group
assignments, and data paths unchanged. Optional digest fields retained in older
manifests are not used by these checks.
