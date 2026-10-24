# Data use and privacy

[English](DATA_USE.md) · [简体中文](DATA_USE_zh.md)

Head2Field distributes source code, documentation, configurations, schemas,
synthetic examples, and pretrained model weights. Participant images,
segmentations, meshes, electrode localizations, FEM outputs, cohort manifests,
labels, predictions, training statistics/caches, logs, and training-resume
states are kept outside the software distribution.

## Software, weights, and datasets

The source code uses [GPL-3.0-only](LICENSE_GUIDE.md). Dataset access and use
remain governed by the upstream terms.

The wheel and source archive include model weights, checksums, a model card,
weight documentation, license information, and data-use notices. The weights
are a bare model state dictionary; this tensor-only format does not replace
the terms governing the training data. See [weights](../weights/README.md).

Obtain imaging data and external scientific dependencies under their
applicable terms. The repository does not supply participant datasets,
templates, atlases, or a SimNIBS installation.

## Keep participant data out of source changes

Use small synthetic inputs in bug reports and contribution examples. Keep
participant images, meshes, FEM labels, predictions, subject/family
identifiers, group manifests, re-identification keys, training-resume bundles,
and credentials out of commits and issue attachments.

Store generated research outputs separately from source code. For local
development checks, use an ignored task directory under `.work/`.
Before sharing a log or example, remove participant identifiers, account
details, and local storage paths.

## Synthetic examples

The generators create deterministic, small, non-anatomical arrays and fictional
identifiers. They do not transform, average, or redistribute participant data.
Use them to check file loading and software workflows; accuracy on participant
data requires a separate evaluation.

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

Users are responsible for dataset access, institutional and ethics
requirements, secure storage, and compliance with the terms of their inputs
and dependencies. The [model card](MODEL_CARD.md) describes the assumptions for
non-clinical research use. The software does not provide participant consent,
dataset access, clinical validation, or calibrated uncertainty estimates.
