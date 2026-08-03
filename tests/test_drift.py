"""Drift detection tests.

Two properties matter more than the arithmetic: **no drift must not fire** (a monitor that
cries wolf gets muted, which is worse than no monitor), and **real drift must fire**.
"""

from __future__ import annotations

import numpy as np
import pytest

from cropguard.monitoring.drift import (
    PSI_SIGNIFICANT,
    TVD_SIGNIFICANT,
    class_distribution_drift,
    detect_drift,
    interpret_psi,
    ks_test,
    population_stability_index,
    total_variation_distance,
)


def _confidence(n, low=0.90, high=1.0, seed=0):
    return np.random.default_rng(seed).uniform(low, high, size=n)


# --- PSI ---------------------------------------------------------------------------------


def test_psi_of_a_distribution_against_itself_is_zero():
    sample = _confidence(5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_near_zero_for_two_draws_from_the_same_distribution():
    """The false-positive case. Two samples of the same process must not look like drift."""
    a = _confidence(5000, seed=1)
    b = _confidence(5000, seed=2)
    assert population_stability_index(a, b) < 0.1


def test_psi_grows_with_the_size_of_the_shift():
    reference = _confidence(5000, seed=3)
    small = population_stability_index(reference, _confidence(5000, 0.85, 0.98, seed=4))
    large = population_stability_index(reference, _confidence(5000, 0.30, 0.60, seed=5))
    assert large > small > 0


def test_psi_flags_a_confidence_collapse():
    """The signature of inputs drifting off-distribution: the model hedges."""
    reference = _confidence(5000, 0.95, 1.0, seed=6)
    current = _confidence(5000, 0.20, 0.60, seed=7)
    psi = population_stability_index(reference, current)
    assert psi > PSI_SIGNIFICANT
    assert interpret_psi(psi) == "significant shift"


def test_psi_is_symmetric_about_mass_appearing_and_vanishing():
    a = _confidence(4000, 0.90, 1.0, seed=8)
    b = _confidence(4000, 0.40, 0.70, seed=9)
    assert population_stability_index(a, b) == pytest.approx(
        population_stability_index(b, a), rel=0.35
    )


def test_psi_survives_a_constant_reference():
    """A degenerate reference collapses every quantile onto one value — must not divide by
    zero or return inf."""
    constant = np.full(500, 0.9)
    assert np.isfinite(population_stability_index(constant, _confidence(500)))


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index(np.array([]), _confidence(10))


def test_interpretation_thresholds():
    assert interpret_psi(0.05) == "stable"
    assert interpret_psi(0.15) == "moderate shift"
    assert interpret_psi(0.40) == "significant shift"


# --- KS ----------------------------------------------------------------------------------


def test_ks_does_not_reject_identical_distributions():
    _, p = ks_test(_confidence(2000, seed=10), _confidence(2000, seed=11))
    assert p > 0.01


def test_ks_rejects_a_shifted_distribution():
    statistic, p = ks_test(
        _confidence(2000, 0.9, 1.0, seed=12), _confidence(2000, 0.3, 0.5, seed=13)
    )
    assert statistic > 0.9
    assert p < 1e-10


# --- class mix ---------------------------------------------------------------------------


def test_class_drift_quiet_when_the_mix_is_stable():
    reference = np.array([500, 300, 200])
    current = np.array([250, 150, 100])  # same proportions, half the volume
    statistic, p = class_distribution_drift(reference, current)
    assert statistic == pytest.approx(0.0, abs=1e-6)
    assert p > 0.9


def test_class_drift_fires_when_the_model_collapses_onto_one_class():
    reference = np.array([500, 300, 200])
    current = np.array([10, 10, 980])
    statistic, p = class_distribution_drift(reference, current)
    assert statistic > 100
    assert p < 1e-10


def test_class_drift_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same length"):
        class_distribution_drift(np.array([1, 2, 3]), np.array([1, 2]))


def test_class_drift_rejects_empty_counts():
    with pytest.raises(ValueError, match="non-empty"):
        class_distribution_drift(np.array([0, 0]), np.array([1, 1]))


# --- the report --------------------------------------------------------------------------


def test_report_stays_quiet_on_a_stable_window():
    reference, current = _confidence(3000, seed=14), _confidence(3000, seed=15)
    report = detect_drift(reference, current)

    assert not report.drift_detected
    assert report.psi_verdict == "stable"
    assert "no actionable drift" in report.summary()


def test_report_fires_and_explains_a_confidence_collapse():
    reference = _confidence(3000, 0.95, 1.0, seed=16)
    current = _confidence(3000, 0.25, 0.55, seed=17)
    report = detect_drift(reference, current)

    assert report.drift_detected
    assert "investigate" in report.summary()
    assert any("mean confidence has fallen" in note for note in report.notes)


def test_report_warns_when_the_window_is_too_small_to_trust():
    report = detect_drift(_confidence(3000, seed=18), _confidence(40, seed=19))
    assert any("only 40 current samples" in note for note in report.notes)


# --- total variation distance ------------------------------------------------------------


def test_tvd_is_zero_for_identical_mixes_regardless_of_volume():
    """The property a p-value lacks: TVD does not move with sample count."""
    assert total_variation_distance(np.array([500, 300, 200]), np.array([50, 30, 20])) == (
        pytest.approx(0.0, abs=1e-9)
    )


def test_tvd_is_one_for_disjoint_mixes():
    assert total_variation_distance(np.array([100, 0]), np.array([0, 100])) == pytest.approx(1.0)


def test_tvd_flags_a_collapse_onto_one_class():
    tvd = total_variation_distance(np.array([500, 300, 200]), np.array([10, 10, 980]))
    assert tvd > TVD_SIGNIFICANT


# --- the report --------------------------------------------------------------------------


def test_class_mix_shift_is_its_own_signal():
    """Confidence steady, class mix collapsed. That is a real event and must surface — the
    earlier design missed it because the verdict keyed on PSI alone."""
    rng = np.random.default_rng(20)
    reference_predictions = rng.integers(0, 5, size=2000)
    current_predictions = np.full(2000, 3)  # everything is now one class

    report = detect_drift(
        _confidence(2000, seed=21),
        _confidence(2000, seed=22),
        reference_predictions,
        current_predictions,
        num_classes=5,
    )
    assert not report.confidence_shifted  # the model is just as sure as before
    assert report.class_mix_shifted  # but it is answering a different question
    assert report.drift_detected
    assert "class mix changed while confidence held" in report.summary()


def test_stable_class_mix_does_not_fire():
    rng = np.random.default_rng(25)
    reference_predictions = rng.integers(0, 5, size=3000)
    current_predictions = rng.integers(0, 5, size=3000)

    report = detect_drift(
        _confidence(3000, seed=26),
        _confidence(3000, seed=27),
        reference_predictions,
        current_predictions,
        num_classes=5,
    )
    assert not report.class_mix_shifted
    assert not report.drift_detected


def test_report_is_json_serializable():
    import json

    report = detect_drift(_confidence(500, seed=23), _confidence(500, seed=24))
    payload = json.loads(json.dumps(report.to_dict()))
    assert "confidence_psi" in payload
    assert payload["drift_detected"] is False
