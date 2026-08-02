"""Calibration tests.

Checked against constructed cases with known answers rather than against our own output — a
calibration metric that only agrees with itself tells you nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from cropguard.evaluation.calibration import (
    TemperatureScaler,
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
    softmax,
)


def _perfectly_calibrated(n=20000, seed=0):
    """Confidence p on binary-style outputs, correct with probability exactly p."""
    rng = np.random.default_rng(seed)
    confidence = rng.uniform(0.5, 1.0, size=n)
    correct = rng.random(n) < confidence
    probabilities = np.zeros((n, 2))
    # Column 0 holds the confident class; make it right exactly `correct` of the time.
    probabilities[:, 0] = confidence
    probabilities[:, 1] = 1 - confidence
    labels = np.where(correct, 0, 1)
    return probabilities, labels


def test_perfect_calibration_gives_near_zero_ece():
    probabilities, labels = _perfectly_calibrated()
    assert expected_calibration_error(probabilities, labels, n_bins=15) < 0.02


def test_overconfident_model_has_large_ece():
    """Always claims 0.99, is right half the time -> ECE should approach |0.99 - 0.5|."""
    n = 2000
    probabilities = np.tile([0.99, 0.01], (n, 1))
    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.49, abs=0.02)


def test_underconfident_model_is_also_penalised():
    """ECE measures |gap|, so under-confidence counts too - which is our actual failure mode."""
    n = 2000
    probabilities = np.tile([0.60, 0.40], (n, 1))
    labels = np.zeros(n, dtype=int)  # always correct despite claiming 0.6

    report = evaluate_calibration(probabilities, labels)
    assert report.ece == pytest.approx(0.40, abs=0.02)
    assert report.gap < 0  # negative gap == under-confident
    assert "under-confident" in report.summary()


def test_mce_is_at_least_ece():
    """The worst bin cannot be better than the weighted average of all bins."""
    probabilities, labels = _perfectly_calibrated(n=5000, seed=3)
    ece = expected_calibration_error(probabilities, labels)
    mce = maximum_calibration_error(probabilities, labels)
    assert mce >= ece


def test_brier_rewards_a_confident_correct_model():
    labels = np.zeros(100, dtype=int)
    confident = np.tile([0.99, 0.01], (100, 1))
    hedged = np.tile([0.55, 0.45], (100, 1))

    assert brier_score(confident, labels) < brier_score(hedged, labels)


def test_brier_catches_what_ece_misses():
    """A model reporting constant confidence equal to its accuracy scores a perfect ECE while
    being useless. Brier is a proper scoring rule and penalises it."""
    n = 2000
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    lazy = np.tile([0.5, 0.5], (n, 1))  # never commits

    assert expected_calibration_error(lazy, labels) == pytest.approx(0.0, abs=0.01)
    assert brier_score(lazy, labels) > 0.4  # but Brier is unimpressed


def test_reliability_curve_shape_and_counts():
    probabilities, labels = _perfectly_calibrated(n=4000, seed=5)
    curve = reliability_curve(probabilities, labels, n_bins=10)

    assert len(curve["bin_center"]) == 10
    assert curve["count"].sum() == len(labels)
    populated = curve["count"] > 0
    # Perfectly calibrated data should track the diagonal.
    assert np.nanmax(np.abs(curve["accuracy"][populated] - curve["confidence"][populated])) < 0.1


# --- temperature scaling ---------------------------------------------------------------


def _logits(n=4000, k=8, seed=0):
    """Logits that are *already calibrated*: labels are drawn from softmax(logits), so that
    distribution is the true posterior and the NLL-optimal temperature is 1.

    This matters. Scaling arbitrary logits by c does not make c the optimal temperature —
    that only follows when the starting point is calibrated, which is what makes the recovery
    tests below meaningful rather than coincidental.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2.0, size=(n, k))
    probabilities = softmax(logits)
    # Vectorised categorical sampling: inverse-CDF over each row.
    draws = rng.random((n, 1))
    labels = (probabilities.cumsum(axis=1) < draws).sum(axis=1).clip(0, k - 1)
    return logits, labels


def test_calibrated_logits_are_actually_calibrated():
    """Guards the fixture itself — if this drifts, every recovery test below is meaningless."""
    logits, labels = _logits(seed=0)
    assert TemperatureScaler().fit(logits, labels).temperature == pytest.approx(1.0, abs=0.25)
    assert expected_calibration_error(softmax(logits), labels) < 0.05


def test_temperature_recovers_a_known_overconfidence():
    """Sharpening calibrated logits by 3x should be corrected by a temperature near 3."""
    logits, labels = _logits(seed=1)
    scaler = TemperatureScaler().fit(logits * 3.0, labels)
    assert scaler.temperature == pytest.approx(3.0, rel=0.3)


def test_temperature_below_one_for_underconfident_logits():
    """Our real failure mode: squashed logits need sharpening, i.e. T < 1."""
    logits, labels = _logits(seed=2)
    scaler = TemperatureScaler().fit(logits * 0.3, labels)
    assert scaler.temperature == pytest.approx(0.3, rel=0.4)
    assert scaler.temperature < 1.0


def test_temperature_scaling_never_changes_predictions():
    """The safety property: dividing by a positive scalar is monotonic, so argmax is fixed.
    This is what makes it safe to apply to a deployed model."""
    logits, labels = _logits(seed=3)
    scaler = TemperatureScaler().fit(logits, labels)

    before = logits.argmax(axis=1)
    after = scaler.transform(logits).argmax(axis=1)
    assert np.array_equal(before, after)


def test_temperature_scaling_improves_ece_on_miscalibrated_logits():
    logits, labels = _logits(n=4000, seed=4)
    miscalibrated = logits * 4.0  # sharply over-confident

    before = expected_calibration_error(softmax(miscalibrated), labels)
    scaler = TemperatureScaler().fit(miscalibrated, labels)
    after = expected_calibration_error(scaler.transform(miscalibrated), labels)

    assert after < before


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="must be fitted"):
        TemperatureScaler().transform(np.zeros((2, 3)))


def test_softmax_temperature_direction():
    logits = np.array([[3.0, 1.0, 0.0]])
    assert softmax(logits, 5.0).max() < softmax(logits, 1.0).max()  # T>1 softens
    assert softmax(logits, 0.2).max() > softmax(logits, 1.0).max()  # T<1 sharpens
    assert softmax(logits, 1.0).sum() == pytest.approx(1.0)
