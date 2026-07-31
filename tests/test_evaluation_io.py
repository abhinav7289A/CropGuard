"""Round-trip and guard-rail tests for the prediction/comparison CLIs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cropguard.evaluation.compare import _assert_same_holdout
from cropguard.evaluation.predict import load_predictions, save_predictions


def _fake_predictions(n=200, num_classes=5, accuracy=0.8, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n)
    correct = rng.random(n) < accuracy
    predictions = np.where(labels, labels, labels)
    predictions = np.where(correct, labels, (labels + 1) % num_classes)
    probabilities = rng.random((n, num_classes)).astype(np.float32)
    return {
        "paths": np.array([f"class{labels[i]}/img{i}.jpg" for i in range(n)]),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "correct": predictions == labels,
    }


def test_predictions_round_trip_through_disk(tmp_path: Path):
    original = _fake_predictions()
    out = tmp_path / "preds.npz"
    save_predictions(original, out, model_version="v1", split="test")

    loaded = load_predictions(out)
    assert np.array_equal(loaded["labels"], original["labels"])
    assert np.array_equal(loaded["correct"], original["correct"])
    assert np.array_equal(loaded["paths"], original["paths"])
    assert loaded["metadata"]["model_version"] == "v1"


def test_comparison_rejects_predictions_over_different_images():
    a = _fake_predictions(n=100, seed=1)
    b = _fake_predictions(n=100, seed=2)
    b["paths"] = np.array([p.replace("img", "other") for p in b["paths"]])

    with pytest.raises(ValueError, match="different images"):
        _assert_same_holdout(a, b)


def test_comparison_rejects_different_holdout_sizes():
    with pytest.raises(ValueError, match="Different holdout sizes"):
        _assert_same_holdout(_fake_predictions(n=100), _fake_predictions(n=50))


def test_comparison_rejects_disagreeing_ground_truth():
    a = _fake_predictions(n=100, seed=1)
    b = _fake_predictions(n=100, seed=1)
    b["labels"] = (b["labels"] + 1) % 5

    with pytest.raises(ValueError, match="labels disagree"):
        _assert_same_holdout(a, b)


def _write_pair(tmp_path: Path, acc_a: float, acc_b: float) -> tuple[Path, Path]:
    """Two prediction files over the identical holdout, differing only in accuracy."""
    rng = np.random.default_rng(0)
    n, num_classes = 3000, 10
    labels = rng.integers(0, num_classes, size=n)
    paths = np.array([f"class{labels[i]}/img{i}.jpg" for i in range(n)])

    files = []
    for accuracy, name in ((acc_a, "a.npz"), (acc_b, "b.npz")):
        correct = rng.random(n) < accuracy
        predictions = np.where(correct, labels, (labels + 1) % num_classes)
        path = tmp_path / name
        save_predictions(
            {
                "paths": paths,
                "labels": labels,
                "predictions": predictions,
                "probabilities": np.zeros((n, num_classes), dtype=np.float32),
                "correct": predictions == labels,
            },
            path,
            model_version=name,
        )
        files.append(path)
    return files[0], files[1]


def _run_compare(baseline: Path, challenger: Path, out: Path):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "cropguard.evaluation.compare",
            "--baseline",
            str(baseline),
            "--challenger",
            str(challenger),
            "--out",
            str(out),
            "--resamples",
            "2000",
        ],
        capture_output=True,
        text=True,
    )


def test_compare_cli_exits_zero_when_the_challenger_wins(tmp_path: Path):
    baseline, challenger = _write_pair(tmp_path, acc_a=0.70, acc_b=0.85)
    out = tmp_path / "report.json"

    proc = _run_compare(baseline, challenger, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "significantly better" in proc.stdout
    assert out.exists()


def test_compare_cli_exits_nonzero_without_a_real_improvement(tmp_path: Path):
    """The gate must fail closed — CI should not promote a model on noise."""
    baseline, challenger = _write_pair(tmp_path, acc_a=0.80, acc_b=0.80)
    out = tmp_path / "report.json"

    proc = _run_compare(baseline, challenger, out)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "no significant improvement" in proc.stdout
