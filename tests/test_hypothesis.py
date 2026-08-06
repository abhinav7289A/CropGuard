"""Hypothesis-testing tests.

Where possible these check against independently known values (textbook McNemar tables,
scipy's own t-test, published power tables) rather than against our own output — a
statistics module that only agrees with itself is worthless.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cropguard.evaluation.hypothesis import (
    bootstrap_accuracy_difference,
    bootstrap_macro_f1_difference,
    compare_models,
    interpret_effect_size,
    macro_f1,
    mcnemar,
    paired_t_test,
    power_analysis,
)


def _correctness(n_both_correct, n_only_a, n_only_b, n_both_wrong):
    """Build two correctness vectors with an exact discordance structure."""
    a = [True] * n_both_correct + [True] * n_only_a + [False] * n_only_b + [False] * n_both_wrong
    b = [True] * n_both_correct + [False] * n_only_a + [True] * n_only_b + [False] * n_both_wrong
    return np.array(a), np.array(b)


def _imbalanced_case():
    """A 900/50/50 problem where accuracy and macro-F1 rank the two models oppositely.

    A ignores both rare classes and rides the majority: accuracy 0.900, macro-F1 0.316.
    B sacrifices majority-class accuracy to find them: accuracy 0.890, macro-F1 0.781.
    This is the real challenger's behaviour in exaggerated form.
    """
    labels = np.array([0] * 900 + [1] * 50 + [2] * 50)

    pred_a = np.concatenate([np.zeros(900), np.zeros(50), np.zeros(50)]).astype(np.int64)

    pred_b = np.concatenate(
        [
            np.repeat([0, 1], [800, 100]),  # 100 of the majority class lost to class 1
            np.repeat([1, 0], [45, 5]),
            np.repeat([2, 0], [45, 5]),
        ]
    ).astype(np.int64)

    return labels, pred_a, pred_b


# --- McNemar -------------------------------------------------------------------------


def test_mcnemar_matches_textbook_chi_squared_value():
    """n01=12, n10=3 -> X^2 = (|12-3|-1)^2/15 = 4.2667, p = 0.0389."""
    a, b = _correctness(n_both_correct=100, n_only_a=12, n_only_b=3, n_both_wrong=50)
    result = mcnemar(a, b, exact=False)

    assert result.only_a_correct == 12
    assert result.only_b_correct == 3
    assert result.statistic == pytest.approx(64 / 15, abs=1e-6)
    assert result.p_value == pytest.approx(float(stats.chi2.sf(64 / 15, 1)), abs=1e-9)
    assert result.p_value == pytest.approx(0.0389, abs=1e-4)


def test_mcnemar_exact_matches_the_binomial_by_hand():
    """discordant=10, min=2 -> p = 2 * P(X<=2 | n=10, p=0.5) = 2*56/1024."""
    a, b = _correctness(n_both_correct=5, n_only_a=2, n_only_b=8, n_both_wrong=5)
    result = mcnemar(a, b, exact=True)

    assert result.method == "exact binomial"
    assert result.p_value == pytest.approx(2 * 56 / 1024, abs=1e-9)


def test_mcnemar_uses_exact_test_when_discordant_pairs_are_few():
    a, b = _correctness(n_both_correct=500, n_only_a=3, n_only_b=8, n_both_wrong=20)
    assert mcnemar(a, b).method == "exact binomial"

    a, b = _correctness(n_both_correct=500, n_only_a=20, n_only_b=40, n_both_wrong=20)
    assert mcnemar(a, b).method.startswith("chi-squared")


def test_mcnemar_ignores_concordant_pairs():
    """Images both models agree on carry no information about which is better."""
    base = mcnemar(*_correctness(10, 20, 40, 10), exact=False)
    padded = mcnemar(*_correctness(9999, 20, 40, 9999), exact=False)

    assert base.statistic == pytest.approx(padded.statistic)
    assert base.p_value == pytest.approx(padded.p_value)


def test_mcnemar_on_identical_models_is_not_significant():
    correct = np.array([True, False, True, True, False])
    result = mcnemar(correct, correct)

    assert result.n_discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mcnemar(np.array([True, False]), np.array([True]))


# --- Paired t-test / effect size ------------------------------------------------------


def test_paired_t_test_agrees_with_scipy():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 0.9, size=38)
    b = a + rng.normal(0.02, 0.03, size=38)

    result = paired_t_test(a, b)
    expected_t, expected_p = stats.ttest_rel(b, a)

    assert result.t_statistic == pytest.approx(float(expected_t))
    assert result.p_value == pytest.approx(float(expected_p))
    assert result.df == 37


def test_paired_t_test_cohens_d_is_mean_over_sd_of_differences():
    a = np.array([0.10, 0.20, 0.30, 0.40])
    b = np.array([0.15, 0.28, 0.33, 0.50])
    differences = b - a

    result = paired_t_test(a, b)
    assert result.cohens_d == pytest.approx(differences.mean() / differences.std(ddof=1))
    assert result.mean_difference == pytest.approx(differences.mean())


def test_paired_t_test_confidence_interval_brackets_the_mean():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.4, 0.8, size=30)
    b = a + 0.05

    result = paired_t_test(a, b)
    assert result.ci_low <= result.mean_difference <= result.ci_high
    assert result.ci_low > 0  # a constant +0.05 shift must exclude zero


def test_paired_t_test_handles_zero_variance_without_dividing_by_zero():
    a = np.array([0.5, 0.6, 0.7])
    result = paired_t_test(a, a)

    assert result.mean_difference == 0.0
    assert result.p_value == 1.0


def test_effect_size_labels_follow_cohens_conventions():
    assert interpret_effect_size(0.1) == "negligible"
    assert interpret_effect_size(0.3) == "small"
    assert interpret_effect_size(0.6) == "medium"
    assert interpret_effect_size(1.2) == "large"
    assert interpret_effect_size(-1.2) == "large"  # magnitude, not direction


# --- Bootstrap ------------------------------------------------------------------------


def test_bootstrap_ci_contains_the_observed_difference():
    rng = np.random.default_rng(2)
    a = rng.random(2000) < 0.80
    b = rng.random(2000) < 0.85

    result = bootstrap_accuracy_difference(a, b, n_resamples=2000, seed=7)
    assert result.ci_low <= result.observed_difference <= result.ci_high
    assert result.observed_difference == pytest.approx(b.mean() - a.mean())


def test_bootstrap_is_deterministic_for_a_given_seed():
    a, b = _correctness(500, 60, 120, 100)
    first = bootstrap_accuracy_difference(a, b, n_resamples=1000, seed=42)
    second = bootstrap_accuracy_difference(a, b, n_resamples=1000, seed=42)

    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
    assert first.ci_low != bootstrap_accuracy_difference(a, b, n_resamples=1000, seed=1).ci_low


def test_bootstrap_ci_includes_zero_for_identical_models():
    correct = np.random.default_rng(3).random(1000) < 0.8
    result = bootstrap_accuracy_difference(correct, correct, n_resamples=1000)

    assert result.observed_difference == 0.0
    assert not result.excludes_zero


def test_bootstrap_ci_excludes_zero_for_a_large_difference():
    a, b = _correctness(n_both_correct=500, n_only_a=10, n_only_b=300, n_both_wrong=190)
    result = bootstrap_accuracy_difference(a, b, n_resamples=2000, seed=5)

    assert result.excludes_zero
    assert result.ci_low > 0


# --- Macro-F1 --------------------------------------------------------------------------


def test_macro_f1_matches_sklearn():
    """Checked against an independent implementation, not against our own arithmetic.

    sklearn averages over the union of true and predicted labels by default, while we average
    over the classes present in the ground truth, so `labels` is passed explicitly to make the
    two definitions coincide.
    """
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(11)
    labels = rng.integers(0, 8, size=1500)
    predictions = np.where(rng.random(1500) < 0.75, labels, rng.integers(0, 8, size=1500))

    expected = f1_score(labels, predictions, average="macro", labels=np.unique(labels))
    assert macro_f1(labels, predictions, n_classes=8) == pytest.approx(expected)


def test_macro_f1_is_one_for_perfect_predictions():
    labels = np.array([0, 1, 2, 2, 1, 0])
    assert macro_f1(labels, labels) == pytest.approx(1.0)


def test_macro_f1_ignores_classes_absent_from_the_ground_truth():
    """A resample that drops a rare class must not be scored as failing it."""
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1])

    # Class 2 exists in the label space but not in this sample; scoring it 0 would drag
    # the mean to 0.667 and make every bootstrap resample of a rare class a penalty.
    assert macro_f1(labels, predictions, n_classes=3) == pytest.approx(1.0)


def test_macro_f1_and_accuracy_can_rank_two_models_oppositely():
    """The reason this statistic exists — the two metrics are not interchangeable."""
    labels, pred_a, pred_b = _imbalanced_case()

    accuracy_a = float((pred_a == labels).mean())
    accuracy_b = float((pred_b == labels).mean())
    macro_a = macro_f1(labels, pred_a, n_classes=3)
    macro_b = macro_f1(labels, pred_b, n_classes=3)

    assert accuracy_a > accuracy_b
    assert macro_b > macro_a


def test_macro_f1_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Shape mismatch"):
        macro_f1(np.array([0, 1, 2]), np.array([0, 1]))


def test_macro_f1_bootstrap_ci_contains_the_observed_difference():
    labels, pred_a, pred_b = _imbalanced_case()
    result = bootstrap_macro_f1_difference(labels, pred_a, pred_b, n_resamples=300, seed=7)

    observed = macro_f1(labels, pred_b, 3) - macro_f1(labels, pred_a, 3)
    assert result.observed_difference == pytest.approx(observed)
    assert result.ci_low <= result.observed_difference <= result.ci_high


def test_macro_f1_bootstrap_excludes_zero_for_a_clear_difference():
    labels, pred_a, pred_b = _imbalanced_case()
    result = bootstrap_macro_f1_difference(labels, pred_a, pred_b, n_resamples=300, seed=5)

    assert result.excludes_zero
    assert result.ci_low > 0


def test_macro_f1_bootstrap_includes_zero_for_identical_models():
    labels, pred_a, _ = _imbalanced_case()
    result = bootstrap_macro_f1_difference(labels, pred_a, pred_a, n_resamples=200, seed=3)

    assert result.observed_difference == 0.0
    assert not result.excludes_zero


def test_macro_f1_bootstrap_is_deterministic_for_a_given_seed():
    labels, pred_a, pred_b = _imbalanced_case()
    first = bootstrap_macro_f1_difference(labels, pred_a, pred_b, n_resamples=200, seed=42)
    second = bootstrap_macro_f1_difference(labels, pred_a, pred_b, n_resamples=200, seed=42)

    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_macro_f1_bootstrap_rejects_mismatched_shapes():
    labels, pred_a, pred_b = _imbalanced_case()
    with pytest.raises(ValueError, match="Shape mismatch"):
        bootstrap_macro_f1_difference(labels, pred_a[:-1], pred_b, n_resamples=10)


# --- Power ----------------------------------------------------------------------------


def test_power_analysis_matches_published_sample_size_for_medium_effect():
    """Standard result: paired t-test, d_z=0.5, alpha=0.05 two-sided -> n=34 for 80% power."""
    result = power_analysis(effect_size=0.5, n=34)

    assert result.power == pytest.approx(0.80, abs=0.02)
    assert 32 <= result.n_for_80_percent_power <= 36


def test_power_analysis_flags_an_underpowered_comparison():
    result = power_analysis(effect_size=0.2, n=10)

    assert result.power < 0.8
    assert not result.is_adequately_powered
    assert result.n_for_80_percent_power > 10
    assert "UNDERPOWERED" in result.summary()


def test_power_increases_with_sample_size_and_effect_size():
    assert power_analysis(0.5, n=100).power > power_analysis(0.5, n=20).power
    assert power_analysis(0.8, n=30).power > power_analysis(0.3, n=30).power


def test_power_of_a_zero_effect_is_the_false_positive_rate():
    result = power_analysis(effect_size=0.0, n=50)
    assert result.power == pytest.approx(0.05)
    assert result.n_for_80_percent_power == -1


# --- End-to-end comparison ------------------------------------------------------------


def test_compare_models_declares_a_clearly_better_challenger():
    a, b = _correctness(n_both_correct=4000, n_only_a=50, n_only_b=400, n_both_wrong=1000)
    labels = np.arange(a.size) % 38

    result = compare_models(a, b, labels, n_resamples=2000)

    assert result.accuracy_b > result.accuracy_a
    assert result.mcnemar.p_value < 0.05
    assert result.bootstrap.excludes_zero
    assert result.challenger_is_better
    assert "significantly better" in result.summary()


def test_compare_models_does_not_declare_victory_on_noise():
    rng = np.random.default_rng(11)
    a = rng.random(3000) < 0.85
    b = rng.random(3000) < 0.85  # same accuracy, independent errors
    labels = np.arange(a.size) % 38

    result = compare_models(a, b, labels, n_resamples=2000)
    assert not result.challenger_is_better


def test_compare_models_will_not_call_a_worse_model_better():
    """A significant McNemar p-value in the *wrong* direction must not read as a win."""
    a, b = _correctness(n_both_correct=4000, n_only_a=400, n_only_b=50, n_both_wrong=1000)

    result = compare_models(a, b, n_resamples=1000)
    assert result.mcnemar.p_value < 0.05
    assert result.accuracy_b < result.accuracy_a
    assert not result.challenger_is_better


def test_compare_models_warns_when_the_evidence_is_thin():
    a, b = _correctness(n_both_correct=1000, n_only_a=2, n_only_b=5, n_both_wrong=100)
    result = compare_models(a, b, n_resamples=500)

    assert any("discordant pairs" in note for note in result.notes)


def test_compare_models_per_class_uses_one_observation_per_class():
    a, b = _correctness(n_both_correct=380, n_only_a=38, n_only_b=76, n_both_wrong=38)
    labels = np.arange(a.size) % 38

    result = compare_models(a, b, labels, n_resamples=500)
    assert result.per_class is not None
    assert result.per_class.df == 37  # 38 classes -> 37 degrees of freedom
    assert result.power is not None


def test_compare_models_rejects_empty_and_mismatched_input():
    with pytest.raises(ValueError, match="Empty holdout"):
        compare_models(np.array([], dtype=bool), np.array([], dtype=bool))
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_models(np.array([True, False]), np.array([True]))


def test_comparison_result_is_json_serializable():
    import json

    a, b = _correctness(500, 30, 60, 100)
    result = compare_models(a, b, np.arange(a.size) % 5, n_resamples=500)
    payload = json.dumps(result.to_dict())

    assert "mcnemar" in payload and "bootstrap" in payload


def test_compare_models_reports_macro_f1_when_predictions_are_supplied():
    labels, pred_a, pred_b = _imbalanced_case()
    result = compare_models(
        pred_a == labels,
        pred_b == labels,
        labels,
        n_resamples=200,
        predictions_a=pred_a,
        predictions_b=pred_b,
    )

    assert result.macro_f1_a == pytest.approx(macro_f1(labels, pred_a, 3))
    assert result.macro_f1_b == pytest.approx(macro_f1(labels, pred_b, 3))
    assert result.macro_f1_bootstrap is not None
    assert "macro-F1" in result.summary()


def test_compare_models_omits_macro_f1_when_predictions_are_absent():
    """Correctness alone cannot give precision, so the statistic must not be faked."""
    a, b = _correctness(500, 30, 60, 100)
    result = compare_models(a, b, np.arange(a.size) % 5, n_resamples=200)

    assert result.macro_f1_a is None
    assert result.macro_f1_bootstrap is None
    assert not result.macro_f1_favours_challenger
    assert "macro-F1" not in result.summary()


def test_compare_models_requires_labels_for_macro_f1():
    labels, pred_a, pred_b = _imbalanced_case()
    with pytest.raises(ValueError, match="labels are required"):
        compare_models(
            pred_a == labels,
            pred_b == labels,
            labels=None,
            n_resamples=10,
            predictions_a=pred_a,
            predictions_b=pred_b,
        )


def test_the_gate_stays_keyed_on_accuracy_even_when_macro_f1_prefers_the_challenger():
    """The promotion rule was fixed in advance; a macro-F1 win must not override it.

    This is the honesty property of the whole module: the challenger is better on the metric
    that suits it, and the gate still declines to promote, because widening the rule after
    seeing the data is choosing the test that gives the answer you want.
    """
    labels, pred_a, pred_b = _imbalanced_case()
    result = compare_models(
        pred_a == labels,
        pred_b == labels,
        labels,
        n_resamples=300,
        predictions_a=pred_a,
        predictions_b=pred_b,
    )

    assert result.macro_f1_favours_challenger
    assert result.accuracy_b < result.accuracy_a
    assert not result.challenger_is_better
    assert "no significant improvement" in result.summary()


def test_compare_models_flags_when_the_two_metrics_disagree():
    labels, pred_a, pred_b = _imbalanced_case()
    result = compare_models(
        pred_a == labels,
        pred_b == labels,
        labels,
        n_resamples=200,
        predictions_a=pred_a,
        predictions_b=pred_b,
    )

    assert any("metrics disagree" in note for note in result.notes)
    assert "macro-F1 up, accuracy down" in result.summary()
