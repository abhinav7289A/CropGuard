"""Error-analysis tests, checked against hand-computable cases."""

from __future__ import annotations

import numpy as np
import pytest

from cropguard.evaluation.errors import (
    confident_mistakes,
    confusion_pairs,
    error_summary,
    per_class_metrics,
    wilson_interval,
)

CLASSES = ["healthy", "blight", "rust", "scab"]


def _onehot(predictions, n_classes=4, confidence=0.9):
    """Probability rows whose argmax is `predictions` at a chosen confidence."""
    probabilities = np.full((len(predictions), n_classes), (1 - confidence) / (n_classes - 1))
    probabilities[np.arange(len(predictions)), predictions] = confidence
    return probabilities


# --- Wilson intervals ------------------------------------------------------------------


def test_wilson_matches_a_known_value():
    """20/24 successes -> roughly [0.64, 0.94]. This is the Potato___healthy case."""
    low, high = wilson_interval(20, 24)
    assert low == pytest.approx(0.646, abs=0.02)
    assert high == pytest.approx(0.942, abs=0.02)


def test_wilson_stays_inside_zero_one_at_extremes():
    """The normal approximation returns a zero-width interval at p=1 and can exceed 1."""
    low, high = wilson_interval(10, 10)
    assert 0.0 <= low <= high <= 1.0
    assert high == pytest.approx(1.0)
    assert low < 0.8, "a perfect 10/10 must not imply near-certainty"

    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0)
    assert 0.0 < high < 1.0


def test_wilson_narrows_as_sample_grows():
    small = wilson_interval(45, 50)
    large = wilson_interval(900, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_handles_empty_class():
    assert wilson_interval(0, 0) == (0.0, 1.0)


# --- per-class metrics -----------------------------------------------------------------


def test_per_class_metrics_hand_computed():
    #                 healthy healthy blight blight rust
    labels = np.array([0, 0, 1, 1, 2])
    predictions = np.array([0, 1, 1, 1, 2])

    metrics = {m.name: m for m in per_class_metrics(labels, predictions, CLASSES)}

    healthy = metrics["healthy"]
    assert healthy.support == 2
    assert healthy.recall == pytest.approx(0.5)  # 1 of 2 caught
    assert healthy.precision == pytest.approx(1.0)  # 1 of 1 predicted was right

    blight = metrics["blight"]
    assert blight.recall == pytest.approx(1.0)  # both caught
    assert blight.precision == pytest.approx(2 / 3)  # 3 predicted, 2 correct

    assert metrics["scab"].support == 0  # absent class must not crash


def test_low_support_classes_are_flagged():
    labels = np.array([0] * 24 + [1] * 500)
    predictions = np.array([0] * 20 + [1] * 4 + [1] * 500)

    metrics = {m.name: m for m in per_class_metrics(labels, predictions, CLASSES)}
    assert not metrics["healthy"].is_reliable  # n=24
    assert metrics["blight"].is_reliable  # n=500
    # The whole point: a 0.833 recall on 24 samples has a very wide interval.
    assert metrics["healthy"].ci_width > 0.25
    assert "low support" in metrics["healthy"].summary()


# --- confusion pairs -------------------------------------------------------------------


def test_confusion_pairs_ranked_by_count():
    labels = np.array([0] * 10 + [1] * 10)
    predictions = np.array([1] * 4 + [0] * 6 + [2] * 2 + [1] * 8)

    pairs = confusion_pairs(labels, predictions, CLASSES)
    assert pairs[0].true_class == "healthy" and pairs[0].predicted_class == "blight"
    assert pairs[0].count == 4
    assert pairs[0].rate == pytest.approx(0.4)
    assert all(pairs[i].count >= pairs[i + 1].count for i in range(len(pairs) - 1))


def test_confusion_pairs_ignore_singletons_by_default():
    labels = np.array([0, 0, 0, 0])
    predictions = np.array([0, 0, 0, 1])  # one stray error
    assert confusion_pairs(labels, predictions, CLASSES, min_count=2) == []


def test_a_perfect_model_has_no_confusion_pairs():
    labels = np.array([0, 1, 2, 3])
    assert confusion_pairs(labels, labels, CLASSES) == []


# --- confident mistakes ----------------------------------------------------------------


def test_confident_mistakes_ranked_by_confidence():
    labels = np.array([0, 0, 0])
    probabilities = np.array(
        [
            [0.10, 0.85, 0.03, 0.02],  # wrong, confident
            [0.40, 0.45, 0.10, 0.05],  # wrong, hesitant
            [0.90, 0.05, 0.03, 0.02],  # correct
        ]
    )

    mistakes = confident_mistakes(probabilities, labels, CLASSES)
    assert len(mistakes) == 2, "correct predictions must not appear"
    assert mistakes[0]["confidence"] > mistakes[1]["confidence"]
    assert mistakes[0]["predicted_class"] == "blight"
    assert mistakes[0]["true_class"] == "healthy"
    assert mistakes[0]["true_class_probability"] == pytest.approx(0.10)


def test_confident_mistakes_carry_paths_when_given():
    labels = np.array([0, 1])
    probabilities = _onehot(np.array([1, 1]))
    paths = np.array(["healthy/a.jpg", "blight/b.jpg"])

    mistakes = confident_mistakes(probabilities, labels, CLASSES, paths=paths)
    assert len(mistakes) == 1
    assert mistakes[0]["path"] == "healthy/a.jpg"


# --- summary ---------------------------------------------------------------------------


def test_error_summary_is_consistent_and_serializable():
    import json

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=400)
    predictions = labels.copy()
    predictions[:40] = (predictions[:40] + 1) % 4  # 10% error rate

    summary = error_summary(_onehot(predictions), labels, CLASSES)
    assert summary["n_images"] == 400
    assert summary["n_errors"] == 40
    assert summary["accuracy"] == pytest.approx(0.9)
    json.dumps(summary)  # must survive a report write
