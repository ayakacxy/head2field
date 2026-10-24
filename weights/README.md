# Model weights

[English](README.md) · [简体中文](README_zh.md)

The pretrained Head2Field v1 weights are included in the package at
`src/head2field/resources/weights/head2field_v1_weights.pth`.
Prediction loads this file by default.

| Field | Value |
| --- | --- |
| Format | Bare string-to-tensor state dictionary |
| Tensors | 208 float32 tensors |
| Scalar values | 2,607,870 |
| File size | 10,497,106 bytes |
| Factory | `create_head2field_v1()` |

The loader checks parameter names, tensor shapes, FP32 dtype, finite values,
and the complete value count before loading the state dictionary strictly.
The file contains only model tensors; optimizer state, training control,
participant identifiers, paths, and random-number states are excluded.

```python
from head2field import load_head2field_v1_weights

model = load_head2field_v1_weights(device="cpu")
```

To use compatible custom weights, pass `--weights custom_v1.pth` to
`head2field-predict` or provide the path to the loader.
Training exports `best_model.pt` as a bare state dictionary.
Complete `train_state_step_*.pt` files are intended for local resume.
Evaluate each newly trained model separately.

When saving or sharing weights, include the model version, architecture,
checksum, license, and [data-use notice](../docs/DATA_USE.md).
Record the software version separately. Changing a training configuration
affects future runs and does not modify existing weight files.
See [inference](../docs/en/inference.md) for the prediction API and
[evaluation](../docs/en/paper_reproduction.md) for comparisons and timing.
