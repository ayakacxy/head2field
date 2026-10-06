# Model weights

[English](README.md) · [简体中文](README_zh.md)

The pretrained Head2Field v1 state dictionary is bundled at
`src/head2field/resources/weights/head2field_v1_weights.pth` and selected by default.

| Field | Value |
| --- | --- |
| Format | Bare string-to-tensor state dictionary |
| Tensors | 208 float32 tensors |
| Scalar values | 2,607,870 |
| File size | 10,497,106 bytes |
| Factory | `create_head2field_v1()` |

The loader checks exact parameter keys, tensor shapes, FP32 dtype, finite values,
and the complete value count, then loads the model strictly. The asset contains
model tensors only, without optimizer, training control, participant identifiers,
paths, or random-number states.

```python
from head2field import load_head2field_v1_weights

model = load_head2field_v1_weights(device="cpu")
```

Pass `--weights custom_v1.pth` to `head2field-predict` or pass the path to the
loader to use compatible custom weights. Training exports `best_model.pt` as a
bare state dictionary; complete `train_state_step_*.pt` files are for local
resume only. Each new training output needs its own evaluation.

Track the software version, model architecture, and weight checksum separately.
Changing a training configuration affects new training runs; existing pretrained
tensors retain their own identity. Keep the model version, architecture, license,
and [data-use notice](../docs/DATA_USE.md) with distributed weights. See
[inference](../docs/en/inference.md) for the prediction API and
[evaluation](../docs/en/paper_reproduction.md) for comparison and timing guidance.
