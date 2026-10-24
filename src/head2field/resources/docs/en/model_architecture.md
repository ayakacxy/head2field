# Model architecture

[English](model_architecture.md) · [简体中文](../zh/model_architecture.md)

## Overview

Head2Field encodes a subject's head anatomy, then predicts an electrode
response at each requested location. Subtracting source and reference responses
produces the 75 source-to-Oz lead fields. Training and inference use the same
model structure and input processing.

This guide describes the network. For source files and tensor dimensions, see
the [code guide](code_guide.md); for intended use and limitations, see the
[model card](../MODEL_CARD.md).

## Input streams

The network takes three inputs:

1. a 12-channel head volume formed from ten tissue indicator channels, normalized
   scalar conductivity, and normalized T1 intensity;
2. query coordinates and local query features for GM/WM voxels; and
3. ordered source and reference electrode coordinates in world-RAS millimetres.

The prepared-subject loader checks the files before constructing these tensors.
Voxel indices locate data in an array; world coordinates locate points in
physical space. The image affine converts between them.

## Network outline

The anatomical encoder takes a 128-cube input through four levels with
16/32/64/128 channels, group normalization, SiLU, and residual bottlenecks.
At the 16-cube bottleneck, two 96-dimensional mixer blocks apply four-head
global axial attention. A 32-channel FPN merges the four scales through
lateral projections and top-down fusion.

At each query point, trilinear sampling retrieves FPN features and features
from the original 1 mm grid. These are combined with six-band Fourier
coordinates relative to the anatomy centroid. Two SwiGLU residual blocks
produce a 256-dimensional query token.

The shared electrode encoder uses a seven-point patch around each electrode
(the center and six axial neighbors at a 6 mm radius), centroid-relative
coordinates, and 128-dimensional subject context. Its output is a
192-dimensional electrode token. Gated absolute-coordinate branches add
information to both query and electrode tokens.

The decoder uses two FiLM-conditioned residual blocks with 128-dimensional
bottlenecks to predict `G(head, query, electrode)`. It receives normalized
coordinate differences, physical displacement, and distance scaled by 128 mm.
Subtracting two electrode responses gives their source-to-reference field:

```text
E(a -> b) = G(a) - G(b)
E(a -> b) = -E(b -> a)
E(a -> c) = E(a -> b) + E(b -> c)
```

The shared response bank gives these antisymmetric and additive relationships,
up to floating-point roundoff. Both electrode roles use the same network;
there are no role embeddings. Full settings are in
`configs/head2field_v1.json`.

During inference, a session caches subject features and prepared electrode
pairs, then evaluates queries in chunks to limit memory use. The source order,
query order, coordinate frame, and output units stay the same for every chunk.

<a id="fixed-v1-factory"></a>

## Creating the v1 model

Use `create_head2field_v1()` to build the network that matches the bundled
weights. Its settings are in [`../configs/head2field_v1.json`](../../configs/head2field_v1.json).
When loading weights, the program requires:

- a non-empty `str -> Tensor` mapping with no wrapper metadata;
- exactly 208 tensors and 2,607,870 scalar values;
- float32 dtype for every tensor;
- exact key names and tensor shapes; and
- strict loading with no missing or unexpected parameters.

If you change the architecture, parameter names, coordinate conventions, or
input processing, give the model a new version and provide matching weights.

## Stable top-level API

The public `head2field` namespace provides:

- `Head2FieldV1`;
- `create_head2field_v1`;
- `Head2FieldInferenceSession`; and
- `load_head2field_v1_weights`.

Preprocessing, FEM, and training have their own submodules and commands.
Their optional dependencies load when those features are called.

## Model versions

The bundled model uses `Head2FieldV1`, `create_head2field_v1()`,
`load_head2field_v1_weights()`, `configs/head2field_v1.json`, and
`head2field_v1_weights.pth`. A new model version should follow the same naming
pattern, with its own configuration and weight loader.

Model, package, and data-format versions are separate. The package version
is `0.1.0`; a `v2` training-statistics schema can still contain statistics used
to train Head2Field v1.
