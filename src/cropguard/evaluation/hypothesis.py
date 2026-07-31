"""Statistical comparison of two classifiers on a shared holdout set.

The question this module answers is not "is model B more accurate?" — that is just arithmetic
— but "is the difference larger than what resampling the same test set would produce by
chance, and is it large enough to matter?" Those are separate questions, so effect size and
power are reported alongside every p-value.

All tests here are *paired*: both models are evaluated on the identical holdout images, which
is strictly more sensitive than comparing two independent accuracy numbers.

    from cropguard.evaluation.hypothesis import compare_models
    result = compare_models(correct_a, correct_b, labels)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats

ALPHA = 0.05

# Cohen's conventional benchmarks. They are rules of thumb, not laws — reported so a reader
# can apply their own judgement rather than reading significance as importance.
EFFECT_THRESHOLDS = ((0.2, "negligible"), (0.5, "small"), (0.8, "medium"))


def interpret_effect_size(d: float) -> str:
    magnitude = abs(d)
    for threshold, label in EFFECT_THRESHOLDS:
        if magnitude < threshold:
            return label
    return "large"


@dataclass
class McNemarResult:
    """Paired test on the *disagreements* between two classifiers."""

    both_correct: int
    only_a_correct: int
    only_b_correct: int
    both_wrong: int
    statistic: float
    p_value: float
    method: str

    @property
    def n_discordant(self) -> int:
        return self.only_a_correct + self.only_b_correct

    def summary(self) -> str:
        direction = "B > A" if self.only_b_correct > self.only_a_correct else "A > B"
        verdict = "significant" if self.p_value < ALPHA else "not significant"
        return (
            f"McNemar ({self.method}): discordant pairs {self.only_a_correct} vs "
            f"{self.only_b_correct} ({direction}), stat={self.statistic:.4f}, "
            f"p={self.p_value:.4g} -> {verdict} at alpha={ALPHA}"
        )


def mcnemar(
    correct_a: np.ndarray, correct_b: np.ndarray, exact: bool | None = None
) -> McNemarResult:
    """McNemar's test on per-image correctness of two models over the same holdout.

    Only the discordant pairs carry information: images both models get right (or both get
    wrong) say nothing about which is better. `exact` defaults to the binomial test when
    there are few discordant pairs, where the chi-squared approximation is unreliable.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError("Expected 1-D arrays of per-image correctness")

    both_correct = int(np.sum(a & b))
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    both_wrong = int(np.sum(~a & ~b))
    discordant = only_a + only_b

    if exact is None:
        exact = discordant < 25

    if discordant == 0:
        # The models are indistinguishable on this set; no evidence either way.
        return McNemarResult(
            both_correct, only_a, only_b, both_wrong, 0.0, 1.0, "no discordant pairs"
        )

    if exact:
        p_value = float(min(1.0, 2.0 * stats.binom.cdf(min(only_a, only_b), discordant, 0.5)))
        statistic = float(min(only_a, only_b))
        method = "exact binomial"
    else:
        # Edwards' continuity correction; without it the test is anti-conservative.
        statistic = float((abs(only_a - only_b) - 1) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "chi-squared (continuity corrected)"

    return McNemarResult(both_correct, only_a, only_b, both_wrong, statistic, p_value, method)


@dataclass
class PairedTTestResult:
    mean_difference: float
    t_statistic: float
    p_value: float
    df: int
    ci_low: float
    ci_high: float
    cohens_d: float
    effect_size_label: str

    def summary(self) -> str:
        verdict = "significant" if self.p_value < ALPHA else "not significant"
        return (
            f"Paired t-test: mean diff={self.mean_difference:+.4f} "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], t={self.t_statistic:.3f}, "
            f"p={self.p_value:.4g} -> {verdict}; d={self.cohens_d:+.3f} ({self.effect_size_label})"
        )


def paired_t_test(
    scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = ALPHA
) -> PairedTTestResult:
    """Paired t-test on per-class (or per-group) scores, B minus A.

    Typically fed per-class accuracy vectors: 38 paired observations rather than 8,000, which
    asks whether the improvement is spread across classes instead of concentrated in the
    common ones.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.size < 2:
        raise ValueError("Need at least 2 paired observations")

    differences = b - a
    n = differences.size
    mean_difference = float(differences.mean())
    # ddof=1: these are sample estimates, not the population.
    std_difference = float(differences.std(ddof=1))

    if std_difference == 0:
        # Every pair moved identically; t is undefined rather than infinite.
        d = 0.0 if mean_difference == 0 else float(np.inf) * np.sign(mean_difference)
        p_value = 1.0 if mean_difference == 0 else 0.0
        return PairedTTestResult(
            mean_difference,
            float("nan"),
            p_value,
            n - 1,
            mean_difference,
            mean_difference,
            d,
            interpret_effect_size(d),
        )

    t_statistic, p_value = stats.ttest_rel(b, a)
    standard_error = std_difference / np.sqrt(n)
    margin = stats.t.ppf(1 - alpha / 2, df=n - 1) * standard_error
    cohens_d = mean_difference / std_difference  # Cohen's d_z for paired designs

    return PairedTTestResult(
        mean_difference=mean_difference,
        t_statistic=float(t_statistic),
        p_value=float(p_value),
        df=n - 1,
        ci_low=mean_difference - margin,
        ci_high=mean_difference + margin,
        cohens_d=float(cohens_d),
        effect_size_label=interpret_effect_size(cohens_d),
    )


@dataclass
class BootstrapResult:
    observed_difference: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence: float
    p_value: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def summary(self) -> str:
        verdict = "excludes 0" if self.excludes_zero else "includes 0"
        return (
            f"Bootstrap ({self.n_resamples} resamples): diff={self.observed_difference:+.4f}, "
            f"{self.confidence:.0%} CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}] -> {verdict}"
        )


def bootstrap_accuracy_difference(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile CI for accuracy(B) - accuracy(A), resampling images with replacement.

    Both models are resampled on the *same* indices each round, preserving the pairing — that
    is what makes the interval narrow enough to be useful.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    n = a.size
    observed = float(b.mean() - a.mean())

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    differences = b[indices].mean(axis=1) - a[indices].mean(axis=1)

    tail = (1 - confidence) / 2
    ci_low, ci_high = np.percentile(differences, [100 * tail, 100 * (1 - tail)])

    # Two-sided bootstrap p-value: how often the resampled difference crosses zero.
    proportion = (
        float(np.mean(differences <= 0)) if observed > 0 else float(np.mean(differences >= 0))
    )
    p_value = min(1.0, 2 * proportion)

    return BootstrapResult(
        observed_difference=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_resamples=n_resamples,
        confidence=confidence,
        p_value=p_value,
    )


@dataclass
class PowerResult:
    effect_size: float
    n: int
    alpha: float
    power: float
    n_for_80_percent_power: int

    @property
    def is_adequately_powered(self) -> bool:
        return self.power >= 0.8

    def summary(self) -> str:
        if self.is_adequately_powered:
            return f"Power={self.power:.3f} at n={self.n} -> adequately powered (>=0.8)"
        return (
            f"Power={self.power:.3f} at n={self.n} -> UNDERPOWERED; "
            f"n={self.n_for_80_percent_power} needed for 0.8 power at d={self.effect_size:.3f}"
        )


def power_analysis(effect_size: float, n: int, alpha: float = ALPHA) -> PowerResult:
    """Achieved power for a two-sided paired t-test, and the n needed for 0.8.

    Run *after* the experiment it describes: a non-significant result from an underpowered
    test means "we could not tell", not "there is no difference".
    """
    if n < 2:
        raise ValueError("Need at least 2 paired observations")

    def power_at(sample_size: int) -> float:
        if effect_size == 0:
            return alpha
        df = sample_size - 1
        noncentrality = abs(effect_size) * np.sqrt(sample_size)
        critical = stats.t.ppf(1 - alpha / 2, df)
        # Both tails of the noncentral t, so the calculation stays correct if the effect
        # points the other way. scipy's nct underflows at df=1 (the first step of the
        # sample-size search); the clipped result is still correct, so mute the warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            upper = stats.nct.sf(critical, df, noncentrality)
            lower = stats.nct.cdf(-critical, df, noncentrality)
        return float(upper + lower)

    power = power_at(n)

    required = n
    if effect_size != 0:
        required = 2
        while required < 100_000 and power_at(required) < 0.8:
            required += 1
    else:
        required = -1  # no sample size detects a zero effect

    return PowerResult(
        effect_size=float(effect_size),
        n=n,
        alpha=alpha,
        power=power,
        n_for_80_percent_power=required,
    )


@dataclass
class ComparisonResult:
    """Everything needed to defend (or retract) a claim that model B beats model A."""

    n_images: int
    accuracy_a: float
    accuracy_b: float
    mcnemar: McNemarResult
    bootstrap: BootstrapResult
    per_class: PairedTTestResult | None = None
    power: PowerResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def challenger_is_better(self) -> bool:
        """Significant on the paired test AND the CI excludes no-difference."""
        return (
            self.mcnemar.p_value < ALPHA
            and self.mcnemar.only_b_correct > self.mcnemar.only_a_correct
            and self.bootstrap.excludes_zero
        )

    def summary(self) -> str:
        lines = [
            f"n={self.n_images} holdout images",
            f"accuracy: A={self.accuracy_a:.4f}  B={self.accuracy_b:.4f}  "
            f"(diff {self.accuracy_b - self.accuracy_a:+.4f})",
            self.mcnemar.summary(),
            self.bootstrap.summary(),
        ]
        if self.per_class is not None:
            lines.append(self.per_class.summary())
        if self.power is not None:
            lines.append(self.power.summary())
        lines.extend(self.notes)
        lines.append(
            "VERDICT: challenger is significantly better"
            if self.challenger_is_better
            else "VERDICT: no significant improvement demonstrated"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def compare_models(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    labels: np.ndarray | None = None,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> ComparisonResult:
    """Run the full battery: McNemar, bootstrap CI, per-class t-test, power analysis.

    `correct_a` / `correct_b` are per-image booleans for the same holdout images in the same
    order. `labels` enables the per-class analysis.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("Empty holdout set")

    mcnemar_result = mcnemar(a, b)
    bootstrap_result = bootstrap_accuracy_difference(a, b, n_resamples=n_resamples, seed=seed)

    per_class_result = None
    power_result = None
    notes: list[str] = []

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != a.shape:
            raise ValueError(f"labels shape {labels.shape} does not match {a.shape}")
        classes = np.unique(labels)
        if classes.size >= 2:
            acc_a = np.array([a[labels == c].mean() for c in classes])
            acc_b = np.array([b[labels == c].mean() for c in classes])
            per_class_result = paired_t_test(acc_a, acc_b)
            if np.isfinite(per_class_result.cohens_d):
                power_result = power_analysis(per_class_result.cohens_d, n=classes.size)
                if not power_result.is_adequately_powered:
                    notes.append(
                        "NOTE: the per-class test is underpowered — a null result here is "
                        "inconclusive, not evidence of equivalence."
                    )
        else:
            notes.append("NOTE: fewer than 2 classes present; per-class analysis skipped.")

    if mcnemar_result.n_discordant < 25:
        notes.append(
            f"NOTE: only {mcnemar_result.n_discordant} discordant pairs — the models rarely "
            f"disagree, so this comparison rests on very little evidence."
        )

    return ComparisonResult(
        n_images=int(a.size),
        accuracy_a=float(a.mean()),
        accuracy_b=float(b.mean()),
        mcnemar=mcnemar_result,
        bootstrap=bootstrap_result,
        per_class=per_class_result,
        power=power_result,
        notes=notes,
    )
