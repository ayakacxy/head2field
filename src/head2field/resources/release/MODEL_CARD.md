# Model card: Head2Field v1

[English](MODEL_CARD.md) · [简体中文](MODEL_CARD_zh.md)

This page describes the model inputs and outputs, training settings, weights,
and intended use. Method references are listed in the
[citation guide](CITATION_GUIDE.md).

## Model summary

`Head2FieldV1` predicts subject-specific TES lead fields from a T1-derived head
representation, electrode positions, and GM/WM query coordinates. It uses
isotropic scalar conductivities and represents each source-to-reference field
as the difference between two shared electrode responses:

```text
E(x; source -> reference) = G(x, source) - G(x, reference)
```

The output is a float32 `[75, Q, 3]` array: 75 source-to-Oz maps at `Q` query
locations, with world-RAS components `[Ex, Ey, Ez]`. Values are in `V/m` for
`+1 mA` at the source and `-1 mA` at Oz. Prediction uses prepared inputs and
compatible weights. FEM labels are used for supervised training.

<a id="versioning-and-input-contract"></a>

## Model version and input requirements

The model version is `v1`. The software version is recorded separately in
`pyproject.toml` and `CITATION.cff`; changing the software does not retrain the
model or automatically change its input and output formats.

The model expects:

- T1 as the only raw imaging modality;
- prepared inputs on a `256 x 256 x 256`, 1 mm RAS grid;
- ten foreground tissue classes plus background, with the
  `simnibs-4.6-isotropic-v1` scalar-conductivity table in `S/m`;
- twelve input channels: ten tissue indicators, conductivity calculated in
  float32 as `sigma_s_per_m / 1.654`, and normalized T1 intensity;
- a 128-cube encoder input produced by the fixed v1 model factory;
- 76 ordered cap electrodes: Oz as reference and 75 ordered sources;
- GM/WM queries in the recorded input order, with voxel/world positions
  converted through the image affine;
- the current convention and output units given above; and
- FP32 weights with the expected parameter names and tensor shapes.

File layouts and validation details are in
[input requirements](en/input_contract.md) and [data formats](en/data_contracts.md).
The [architecture guide](en/model_architecture.md) describes the network.

## Training

The default real-data configuration uses AdamW, an initial learning rate of
`2e-4`, weight decay of `1e-4`, CUDA FP16 AMP, and gradient norm clipping at 1.0.
Each step processes one subject, eight source-to-Oz pairs, and a chunk of
49,152 GM/WM query locations.

An epoch covers every source-batch/query-chunk combination for each training
subject. Training runs for at most 25 epochs with a 25-epoch cosine schedule.
Epoch-end validation uses the combined data and physics objective with
`a(t)=1`. Training stops after five consecutive epochs without improvement;
test evaluation uses the checkpoint with the best validation result.

The data loss uses 50/100 mm radial regions, per-step scaling
`M / (75 * N_train)`, and statistics in `training_loss_statistics_v2` format.
It is combined with circulation, WM/GM normal-current continuity,
tangential-field continuity, and flux losses. Physics thresholds are fitted
only on training data. The run settings record group-disjoint training,
validation, and test partitions and their data paths.

See [training](en/training.md) for preparation, loss settings, and resume steps.

<a id="verified-weight-format"></a>

## Bundled weight format

| Field | Value |
| --- | --- |
| State dictionary | Bare string-to-tensor mapping |
| Tensors | 208 contiguous float32 tensors |
| Scalar values | 2,607,870 |
| File size | 10,497,106 bytes |

The file stores model tensors only. It contains no optimizer, scheduler,
scaler, step, epoch, metric, manifest, participant identifier, path, sampler,
or RNG state. Parameter names and tensor shapes match the v1 factory.
The wheel and source distribution include these weights and load them by
default. See [weights](../weights/README.md) and [data use](DATA_USE.md).

## Intended use

Head2Field supports TES field modeling, method comparisons, and research on
individualized stimulation planning. The predicted basis can be reused for
tDCS field synthesis and TIS envelope prediction. The manuscript's TIS
optimization experiments use this basis in a separate downstream optimizer.
The package provides basis prediction and a Python query API for these uses.

## Limitations

- The evaluated cohort comprised 200 healthy adults aged 22–36 years from
  WU-Minn HCP. Training, validation, and test partitions contained 128, 32,
  and 40 subjects, with families kept together.
- Tissue conductivities were fixed and isotropic.
- Electrodes had a fixed cylindrical geometry: 10 mm diameter and 4 mm
  thickness. Position-generalization experiments tested 5–20 mm displacements.
- The model returns field vectors; outputs do not include uncertainty estimates.

The manuscript identifies broader age and patient cohorts, anisotropic
conductivity, and variable electrode geometry as directions for further study.

## Reproducibility and evaluation

The manuscript reports a mean full-field relative L2 error of 0.070 for the
128-subject model, compared with 0.078 without physics regularization. Errors
at the tested unseen electrode positions were 0.069–0.070 without fine-tuning.
TIS plans optimized using the predicted basis achieved 94.9% of FEM-optimized
focality under FEM evaluation. The [evaluation guide](en/paper_reproduction.md#manuscript-results)
records the cohort, comparisons, and runtime conditions.

Synthetic tests cover weight/schema loading, coordinates, source and query
ordering, preprocessing/FEM adapters, complete epochs, learning-rate
scheduling, validation, early stopping, and local resume. CUDA FP16 operation
has also been exercised with synthetic inputs.

Synthetic tests exercise software behavior. The manuscript's field-accuracy
results use the cohort data, trained models, and FEM comparisons described
in the evaluation guide.

## Data and acknowledgements

Source and Python distributions exclude participant-level data and complete
training-resume states. Development benefited from WU-Minn HCP data.
[Data use](DATA_USE.md) lists the upstream terms and acknowledgement references.
The code license, dataset terms, and weight notices apply to their respective
materials.
