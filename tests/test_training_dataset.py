"""Isolation and integrity tests for a two-fictional-subject synthetic training dataset."""

from __future__ import annotations

import json

import pytest

from head2field.training.synthetic import (
    SYNTHETIC_SOURCE_NAMES,
    load_synthetic_training_dataset,
    write_synthetic_training_dataset,
)


def test_synthetic_dataset_has_two_group_disjoint_subjects(tmp_path) -> None:
    manifest = write_synthetic_training_dataset(tmp_path / "synthetic")
    subjects = load_synthetic_training_dataset(manifest)

    assert len(subjects) == 2
    assert {subject.partition for subject in subjects} == {"train", "validation"}
    assert len({subject.subject_key for subject in subjects}) == 2
    assert len(SYNTHETIC_SOURCE_NAMES) == 75
    assert "Oz" not in SYNTHETIC_SOURCE_NAMES
    assert all(subject.targets.shape == (8, 75, 3) for subject in subjects)
