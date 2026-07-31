from __future__ import annotations

from collections import Counter
from pathlib import Path

from cropguard.data.split import collect_samples, stratified_split

from .conftest import CLASSES

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_collect_samples_uses_posix_relative_paths(tiny_dataset: Path):
    paths, labels, class_names = collect_samples(tiny_dataset)
    assert class_names == sorted(CLASSES)
    assert len(paths) == len(labels) == 18
    assert all("\\" not in p for p in paths)
    # Label index must agree with the sorted class-name order used by classes.json.
    assert all(
        class_names[label] == path.split("/")[0] for path, label in zip(paths, labels, strict=True)
    )


def test_splits_are_disjoint_and_exhaustive(tiny_dataset: Path):
    paths, labels, _ = collect_samples(tiny_dataset)
    splits = stratified_split(paths, labels, FRACTIONS, seed=42)

    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train | val | test == set(paths)
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) + len(val) + len(test) == len(paths)


def test_split_is_deterministic_for_a_given_seed(tiny_dataset: Path):
    paths, labels, _ = collect_samples(tiny_dataset)
    first = stratified_split(paths, labels, FRACTIONS, seed=42)
    second = stratified_split(paths, labels, FRACTIONS, seed=42)
    assert first == second

    different = stratified_split(paths, labels, FRACTIONS, seed=7)
    assert different != first


def test_every_class_appears_in_every_split(tiny_dataset: Path):
    paths, labels, _ = collect_samples(tiny_dataset)
    splits = stratified_split(paths, labels, FRACTIONS, seed=42)
    for name, split_paths in splits.items():
        classes_present = Counter(p.split("/")[0] for p in split_paths)
        assert set(classes_present) == set(CLASSES), f"{name} split lost a class"
