# Model card: Head2Field v1

[English](MODEL_CARD.md) · [简体中文](MODEL_CARD_zh.md)

This package provides the **official implementation** of Head2Field, described
in our paper:

> **Head2Field: A Continuous Neural Field for Subject-Specific Multi-Electrode
> Transcranial Electrical Stimulation Lead-Field Prediction from Structural
> MRI**

## Model summary

`Head2FieldV1` is a PyTorch model for subject-specific TES electric
lead-field prediction. It combines a T1-derived head representation with
electrode and GM/WM query coordinates. The v1 model uses isotropic scalar
conductivities and the relation

```text
E(x; source -> reference) = G(x, source) - G(x, reference)
```

The model predicts 75 source-to-Oz maps. Its output is a float32 `[75, Q, 3]`
array with world-RAS components `[Ex, Ey, Ez]`, in `V/m` for `+1 mA` at each
source and `-1 mA` at Oz. Inference requires prepared inputs and compatible
weights; FEM labels are needed for supervised training, not prediction.

## Versioning and input contract

The model-contract version is `v1`; the software package version is recorded in
`pyproject.toml` and `CITATION.cff`. A software patch does not by itself create a
new trained model or change the input/output contract.

The v1 contract fixes:

- T1 as the only raw imaging modality;
- a canonical `256 x 256 x 256`, 1 mm RAS prepared-input grid;
- ten foreground tissue classes plus background and the
  `simnibs-4.6-isotropic-v1` scalar-conductivity table in `S/m`;
- twelve model-input channels: ten tissue indicators, conductivity formed in
  float32 as `sigma_s_per_m / 1.654`, and normalized T1 intensity;
- the fixed v1 model factory, using a 128-cube encoder input;
- 76 ordered cap electrodes, with Oz as reference and 75 ordered sources;
- exact GM/WM query ordering and explicit voxel/world-coordinate conversion;
- the current convention and output units stated above; and
- strict FP32 weight loading with fixed parameter keys and tensor shapes.

See [input contracts](en/input_contract.md),
[architecture](en/model_architecture.md), and
[data contracts](en/data_contracts.md) for the machine-readable details.

## Training

The executable real-data profile uses AdamW with initial learning rate `2e-4`
and weight decay `1e-4`. Each step processes one subject, eight source-to-Oz
pairs, and query chunks of 49,152 GM/WM locations. Training uses CUDA FP16 AMP
and gradient norm clipping at 1.0.

Each epoch covers the complete source-batch/query-chunk combinations for every
training subject. The maximum budget is 25 epochs, with a cosine schedule over
25 epochs. Validation runs at epoch end with the combined data/physics objective
and `a(t)=1`. Five consecutive non-improving validation epochs stop training;
the best validation checkpoint is selected for test evaluation.

The data objective uses 50/100 mm radial regions and fixed per-step scaling
`M / (75 * N_train)`. Its statistics contract is `training_loss_statistics_v2`.

The objective combines stable-energy data loss with circulation, WM/GM
normal-current continuity, tangential-field continuity, and flux terms. Physics
thresholds are fitted on training data. Group-disjoint train/validation/test
partitions and asset paths are part of the run contract. See
[training](en/training.md) for loss settings, asset preparation, and resume.

## Verified weight format

The reference v1 weight file contains:

| Field | Value |
| --- | --- |
| State dictionary | Bare string-to-tensor mapping |
| Tensors | 208 contiguous float32 tensors |
| Scalar values | 2,607,870 |
| File size | 10,497,106 bytes |

It contains no optimizer, scheduler, scaler, step, epoch, metric, manifest,
participant identifier, path, sampler, or RNG state. Tensor keys and shapes
match the v1 factory. These pretrained weights are bundled with the wheel and
source distribution and selected by default; see
[weight documentation](../weights/README.md) and [data use](DATA_USE.md).

## Intended use

The model supports non-clinical research, software validation, method
comparison, and reproducible TES field-modeling experiments on inputs that meet
the v1 contract. Users should understand neuroimaging spatial metadata,
electrode/current conventions, and numerical validation of learned surrogates.

The package predicts a lead-field basis. It does not perform stimulation-current
optimization, treatment planning, or uncertainty quantification.

## Limitations

- Invalid tissue labels, conductivities, affines, electrode order, query order,
  or units can invalidate predictions even when tensor shapes are correct.
- The model uses fixed isotropic scalar conductivities. Anisotropy or changed
  model/input semantics requires a different model contract.
- Training data do not establish performance for every scanner, imaging
  protocol, age group, pathology, lesion, implant, or unusual anatomy.
- Outputs have no calibrated uncertainty estimate.
- Predictions have not been validated for diagnosis, treatment selection,
  patient safety decisions, or autonomous clinical control.

Subject-appropriate FEM comparison remains necessary when an experiment depends
on accuracy outside the established evaluation conditions.

## Reproducibility and evaluation

The repository includes synthetic tests for weight/schema loading, coordinate
and source/query contracts, preprocessing/FEM adapters, full epoch coverage,
learning-rate scheduling, validation, early stopping, and local resume.
CUDA FP16 operation has also been exercised on synthetic inputs.

These checks establish software behavior and compatibility. They do not measure
cohort accuracy, generalization, clinical validity, or end-to-end equivalence to
FEM. Reproducing a scientific result requires the corresponding data split,
input preparation, model asset, evaluation protocol, and lawful data access.

## Data and acknowledgements

Participant-level data and training-resume states are excluded from source and
Python distributions. Model development benefited from WU-Minn HCP data; the
[data-use document](DATA_USE.md) provides upstream terms and acknowledgement
references. The source-code license, dataset terms, and any weight-asset notice
apply to their respective materials.
