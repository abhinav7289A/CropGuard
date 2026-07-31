from __future__ import annotations

from pathlib import Path

from cropguard.data.validate import validate_imagefolder

from .conftest import CLASSES, write_image


def test_clean_dataset_passes(tiny_dataset: Path):
    report = validate_imagefolder(tiny_dataset, min_resolution=128, min_samples_per_class=6)
    assert report.passed
    assert report.total_images == 18
    assert set(report.class_counts) == set(CLASSES)
    assert not report.corrupt and not report.low_resolution


def test_corrupt_image_is_flagged_and_fails(tiny_dataset: Path):
    (tiny_dataset / CLASSES[0] / "broken.jpg").write_bytes(b"not an image")
    report = validate_imagefolder(tiny_dataset, min_resolution=128, min_samples_per_class=6)
    assert not report.passed
    assert any("broken.jpg" in p for p in report.corrupt)


def test_low_resolution_image_is_excluded_from_class_count(tiny_dataset: Path):
    write_image(tiny_dataset / CLASSES[1] / "small.jpg", size=(64, 64))
    report = validate_imagefolder(tiny_dataset, min_resolution=128, min_samples_per_class=6)
    assert len(report.low_resolution) == 1
    # The undersized image must not count toward the class minimum.
    assert report.class_counts[CLASSES[1]] == 6
    assert report.total_images == 19


def test_underpopulated_class_fails_validation(tiny_dataset: Path):
    report = validate_imagefolder(tiny_dataset, min_resolution=128, min_samples_per_class=100)
    assert not report.passed
    assert set(report.underpopulated_classes) == set(CLASSES)
    assert report.to_dict()["passed"] is False


def test_non_image_files_are_ignored(tiny_dataset: Path):
    (tiny_dataset / CLASSES[0] / "manifest.json").write_text("{}", encoding="utf-8")
    report = validate_imagefolder(tiny_dataset, min_resolution=128, min_samples_per_class=6)
    assert report.passed
    assert report.total_images == 18
