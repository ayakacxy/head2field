# Data use and privacy

[English](DATA_USE.md) · [简体中文](DATA_USE_zh.md)

The software distribution contains source code, documentation, configurations,
schemas, synthetic examples, and pretrained v1 model weights. It excludes
participant images, segmentations,
meshes, electrode localizations, FEM outputs, cohort manifests, labels,
predictions, training statistics/caches, logs, and training-resume states.

## Software, weights, and datasets

The software is licensed under [GPL-3.0-only](LICENSE_GUIDE.md). This code license does
not grant access to upstream datasets or replace their data-use terms.

The wheel and source archive include the v1 model weights, checksums, model
card, weight documentation, license information, and data-use notices. The
weights are a bare model state dictionary; a tensor-only format does not replace
the terms governing the training data. See [weights](../weights/README.md).

Users obtain their own imaging data and install external scientific dependencies
under the applicable terms. No participant dataset, template, atlas, or SimNIBS
installation is supplied by this repository.

## Keep participant data out of source changes

Do not commit or attach participant-level images, meshes, FEM labels,
predictions, subject/family identifiers, group manifests, re-identification
keys, training-resume bundles, or credentials to code changes or issue reports.
Use small synthetic examples to reproduce software problems.

Generated research outputs belong in a separate data location. Local development
checks can use an ignored task directory under `.work/`. Before sharing logs or
examples, remove participant identifiers, account details, and local storage
paths.

## Synthetic examples

The included generators create deterministic, small, non-anatomical arrays and
fictional identifiers. They do not transform, average, or redistribute data from
real participants. Their purpose is to verify software contracts and workflows,
not to establish model accuracy or biological validity.

## HCP acknowledgement and data-use terms

Development of this method benefited from data provided by the Human Connectome
Project, WU-Minn Consortium. We acknowledge the consortium's acquisition and
processing work and the supporting NIH and McDonnell Center resources.

HCP users, including people accessing shared copies, must accept the applicable
data-use terms. Those terms address participant privacy, institutional research
requirements, distribution of original/derived data, acknowledgements, and
methods citations:

- [HCP Young Adult Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms)
- [WU-Minn HCP Open Access Data Use Terms and prescribed acknowledgement](https://www.humanconnectome.org/storage/app/media/data_use_terms/DataUseTerms-HCP-Open-Access-26Apr2013.pdf)

For publications and presentations using HCP data, follow the prescribed
acknowledgement wording in the official terms and cite the relevant acquisition
and processing methods. Apply the upstream terms to any original or derived
data being shared; this document does not decide the legal classification of a
trained weight file.

## Research use

Users are responsible for dataset access, institutional and ethics requirements,
secure storage, and compliance with the terms of their inputs and dependencies.
The model supports non-clinical research under the assumptions described in
[MODEL_CARD.md](MODEL_CARD.md). It does not supply participant consent, dataset
access, clinical validation, or calibrated uncertainty estimates.
