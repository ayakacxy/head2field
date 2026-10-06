# Model architecture

[English](model_architecture.md) · [简体中文](../zh/model_architecture.md)

## Overview

Head2Field v1 combines anatomical encoding, query features, and electrode
geometry to predict the 75 source-to-Oz lead fields. Its fixed factory and strict
weight schema keep the architecture consistent across training and inference.
The architecture is described below; intended uses and limitations are in
[MODEL_CARD.md](../MODEL_CARD.md).
The [code guide](code_guide.md) maps each stage to its implementation and lists
the tensor dimensions used throughout the network.

## Input streams

The model combines three kinds of information:

1. a 12-channel head volume formed from ten tissue indicator channels, normalized
   scalar conductivity, and normalized T1 intensity;
2. query coordinates and local query features for GM/WM voxels; and
3. ordered source and reference electrode coordinates in world-RAS millimetres.

The prepared-subject loader validates the source data before these tensors are
constructed. Voxel indices and world coordinates are separate representations, converted
using the affine.

## Network outline

The 128-cube encoder has four levels with 16/32/64/128 channels, group
normalization, SiLU, and residual bottlenecks. Two 96-dimensional mixer blocks
use four-head global axial attention at the 16-cube bottleneck. A 32-channel FPN
combines all four scales through lateral projections and top-down fusion.

Query tokens combine trilinearly sampled FPN features, original 1 mm input
features and six-band centroid-relative Fourier coordinates. Two SwiGLU
residual blocks produce 256-dimensional tokens. A shared electrode encoder
combines seven-point patches (center and six axial neighbors, 6 mm radius),
centroid-relative coordinates and 128-dimensional subject context into
192-dimensional electrode tokens. Gated absolute-coordinate branches supplement
both token types.

Two FiLM-conditioned residual blocks with 128-dimensional bottlenecks decode
the shared electrode response `G(head, query, electrode)`. Relative geometry
includes normalized coordinate differences, physical displacement and distance
scaled by 128 mm. Source-to-reference fields are response differences:

```text
E(a -> b) = G(a) - G(b)
E(a -> b) = -E(b -> a)
E(a -> c) = E(a -> b) + E(b -> c)
```

These identities follow from the shared response bank, with floating-point
roundoff in their numerical evaluation. There are no separate source/reference
response networks or role embeddings. The exact settings are recorded in
`configs/head2field_v1.json`.

The inference session caches subject features and electrode-pair preparation so
queries can be evaluated in bounded chunks. Chunking changes memory use but must
not change source order, query order, coordinate frame, or output units.

<a id="fixed-v1-factory"></a>

## Creating the v1 model

`create_head2field_v1()` creates the v1 model used by the bundled weights.
Its architecture settings are listed in [`../configs/head2field_v1.json`](../../configs/head2field_v1.json).
The weight loader additionally checks:

- a non-empty `str -> Tensor` mapping with no wrapper metadata;
- exactly 208 tensors and 2,607,870 scalar values;
- float32 dtype for every tensor;
- exact key names and tensor shapes; and
- strict loading with no missing or unexpected parameters.

Changing the architecture, parameter keys, coordinates, or input processing
requires a new model version and compatible weights.

## Stable top-level API

The `head2field` namespace exports only:

- `Head2FieldV1`;
- `create_head2field_v1`;
- `Head2FieldInferenceSession`; and
- `load_head2field_v1_weights`.

Use the corresponding submodules or commands for preprocessing, FEM, and
training. Optional dependencies are loaded only when those features are used.

## Model versions

The current model uses the versioned names: `Head2FieldV1`,
`create_head2field_v1()`, `load_head2field_v1_weights()`,
`configs/head2field_v1.json`, and `head2field_v1_weights.pth` describe the current
model. Future model versions follow the same naming pattern with their own
architecture configuration and weight loader.

The package version (`0.1.0`) and data-schema versions are independent of the
model version. For example, a `v2` training-statistics schema still describes
data used to train Head2Field v1.
