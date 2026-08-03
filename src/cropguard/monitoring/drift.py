"""Drift detection — noticing a model has gone stale *without* ground-truth labels.

The central constraint of production ML monitoring: **you cannot measure accuracy in
production.** Nobody tells the API which disease the leaf actually had. Test-set metrics
describe the day the model was built and say nothing about today.

What *is* observable is distribution, and three of them are worth watching:

| Signal | Detects | Latency |
|---|---|---|
| Prediction distribution | class balance shifting (new season, new region) | immediate |
| Confidence distribution | inputs drifting away from training data | immediate |
| Feedback accuracy | genuine degradation | slow, needs labels |

Confidence drift is the most useful of the three. It needs no labels, and a model pushed
off-distribution reacts by spreading its probability mass — an effect visible long before
anyone notices wrong answers.

**A caveat this module cannot fix:** none of these prove the model got *worse*. Input drift
means "the world moved", which may or may not hurt. They are triggers to go and look, not
verdicts — which is why the thresholds here open an investigation rather than roll back a
deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# Population Stability Index conventions from credit-risk modelling, where PSI originates.
# They are rules of thumb, not derived from a null distribution — reported so a reader can
# apply their own judgement rather than treating 0.25 as a law.
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25

# Total-variation distance between two class mixes. Bounded [0, 1]: 0 is identical, 1 is
# disjoint. Used instead of the chi-squared p-value because at production volume that p-value
# is meaningless — measured on this project, two samples of the *same* distribution gave
# p = 5.7e-256 purely from n.
TVD_SIGNIFICANT = 0.20


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, epsilon: float = 1e-6
) -> float:
    """PSI between a reference and a current distribution.

        PSI = Σ_b (current_b − reference_b) · ln(current_b / reference_b)

    Symmetric in a useful way: it penalises mass appearing where there was none *and* mass
    vanishing from where there was some. That matters here — a class quietly disappearing from
    production traffic is as interesting as a new one showing up.

    Bin edges come from the **reference** quantiles, not the pooled data. Using pooled edges
    would let the current distribution redefine the yardstick it is being measured against,
    which quietly hides exactly the shift being looked for.

    `epsilon` floors empty bins: `ln(0)` is undefined, and an unbounded PSI from one empty bin
    would swamp the signal.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    current = np.asarray(current, dtype=np.float64).ravel()
    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI needs a non-empty reference and current sample")

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    # Degenerate reference (near-constant) collapses quantiles onto each other.
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0

    reference_share = np.histogram(reference, bins=edges)[0] / reference.size
    current_share = np.histogram(current, bins=edges)[0] / current.size

    reference_share = np.clip(reference_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)

    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def interpret_psi(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate shift"
    return "significant shift"


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov: max gap between the two empirical CDFs.

    Complements PSI rather than duplicating it. KS is binning-free and comes with a p-value,
    but that p-value is a trap at production volume: with 10,000 requests it flags shifts far
    too small to act on. Read the **statistic** as the effect size and the p-value only as a
    floor.
    """
    result = stats.ks_2samp(np.asarray(reference).ravel(), np.asarray(current).ravel())
    return float(result.statistic), float(result.pvalue)


def total_variation_distance(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    """Half the L1 distance between two class distributions. Bounded [0, 1].

    This is the number to act on, not the chi-squared p-value. TVD is an **effect size**: it
    does not grow with sample count, so it means the same thing on 200 requests and 200,000.
    A p-value does the opposite, and a monitor whose sensitivity depends on traffic volume is
    a monitor that will page you at 3am for nothing.
    """
    reference_counts = np.asarray(reference_counts, dtype=np.float64)
    current_counts = np.asarray(current_counts, dtype=np.float64)
    if reference_counts.shape != current_counts.shape:
        raise ValueError("class count vectors must have the same length")
    if reference_counts.sum() == 0 or current_counts.sum() == 0:
        raise ValueError("both count vectors must be non-empty")

    reference_share = reference_counts / reference_counts.sum()
    current_share = current_counts / current_counts.sum()
    return float(0.5 * np.abs(current_share - reference_share).sum())


def class_distribution_drift(
    reference_counts: np.ndarray, current_counts: np.ndarray
) -> tuple[float, float]:
    """Chi-squared test on predicted-class counts.

    Answers a different question from PSI: not "did confidence shift" but "is the model
    predicting a different *mix* of classes". A model that starts calling everything
    `Tomato___healthy` looks fine on confidence and badly wrong here.
    """
    reference_counts = np.asarray(reference_counts, dtype=np.float64)
    current_counts = np.asarray(current_counts, dtype=np.float64)
    if reference_counts.shape != current_counts.shape:
        raise ValueError("class count vectors must have the same length")

    total_reference = reference_counts.sum()
    total_current = current_counts.sum()
    if total_reference == 0 or total_current == 0:
        raise ValueError("both count vectors must be non-empty")

    # Expected current counts if the reference proportions still held.
    expected = reference_counts / total_reference * total_current
    keep = expected > 0
    if keep.sum() < 2:
        return 0.0, 1.0

    statistic = float(np.sum((current_counts[keep] - expected[keep]) ** 2 / expected[keep]))
    p_value = float(stats.chi2.sf(statistic, df=int(keep.sum()) - 1))
    return statistic, p_value


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    confidence_psi: float
    confidence_ks_statistic: float
    confidence_ks_pvalue: float
    class_chi2: float
    class_chi2_pvalue: float
    class_tvd: float
    mean_confidence_reference: float
    mean_confidence_current: float
    notes: list[str] = field(default_factory=list)

    @property
    def psi_verdict(self) -> str:
        return interpret_psi(self.confidence_psi)

    @property
    def confidence_shifted(self) -> bool:
        """Inputs moving away from the training distribution — the model starts hedging."""
        return self.confidence_psi >= PSI_SIGNIFICANT

    @property
    def class_mix_shifted(self) -> bool:
        """The population being served changed, even if confidence did not.

        Judged on TVD rather than the chi-squared p-value: the p-value is a function of traffic
        volume and goes significant on shifts far too small to act on.
        """
        return bool(np.isfinite(self.class_tvd) and self.class_tvd >= TVD_SIGNIFICANT)

    @property
    def drift_detected(self) -> bool:
        """Either signal is worth a look, and they mean different things.

        Confidence drift suggests the model is struggling. Class-mix drift suggests the world
        changed — new season, new region, a client sending only one crop. Neither proves the
        model got *worse*; both are reasons to go and check.
        """
        return self.confidence_shifted or self.class_mix_shifted

    def summary(self) -> str:
        lines = [
            f"reference n={self.n_reference}  current n={self.n_current}",
            f"confidence PSI  {self.confidence_psi:.4f}  ({self.psi_verdict})",
            f"confidence KS   statistic={self.confidence_ks_statistic:.4f} "
            f"p={self.confidence_ks_pvalue:.4g}",
            f"class mix TVD  {self.class_tvd:.4f}"
            f"{'  (SHIFTED)' if self.class_mix_shifted else ''}"
            f"   [chi2={self.class_chi2:.1f} p={self.class_chi2_pvalue:.4g} - ignore the p]",
            f"mean confidence {self.mean_confidence_reference:.4f} -> "
            f"{self.mean_confidence_current:.4f}",
        ]
        lines.extend(self.notes)
        if self.confidence_shifted and self.class_mix_shifted:
            verdict = "VERDICT: investigate — confidence AND class mix have shifted"
        elif self.confidence_shifted:
            verdict = (
                "VERDICT: investigate — confidence has shifted; inputs may be off-distribution"
            )
        elif self.class_mix_shifted:
            verdict = (
                "VERDICT: investigate — the class mix changed while confidence held. "
                "The population being served moved, not necessarily the model's competence."
            )
        else:
            verdict = "VERDICT: no actionable drift"
        lines.append(verdict)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in self.__dict__.items()},
            "psi_verdict": self.psi_verdict,
            "drift_detected": self.drift_detected,
        }


def detect_drift(
    reference_confidence: np.ndarray,
    current_confidence: np.ndarray,
    reference_predictions: np.ndarray | None = None,
    current_predictions: np.ndarray | None = None,
    num_classes: int | None = None,
    n_bins: int = 10,
) -> DriftReport:
    """Compare a production window against a reference window.

    The reference is normally the confidence distribution over the *test* split, captured when
    the model was released — it is the closest thing available to "how this model behaves on
    data it was validated for".

    **The reference and current windows must be sampled the same way.** This bit us: a
    class-balanced validation sample compared against a naturally-imbalanced test split
    reported a chi-squared p of 5.7e-256 — entirely an artefact of the sampling, not drift.
    A reference drawn differently from production makes every comparison meaningless.
    """
    reference_confidence = np.asarray(reference_confidence).ravel()
    current_confidence = np.asarray(current_confidence).ravel()

    psi = population_stability_index(reference_confidence, current_confidence, n_bins)
    ks_statistic, ks_p = ks_test(reference_confidence, current_confidence)

    chi2 = chi2_p = tvd = float("nan")
    notes: list[str] = []
    if reference_predictions is not None and current_predictions is not None:
        k = num_classes or int(max(reference_predictions.max(), current_predictions.max()) + 1)
        reference_counts = np.bincount(np.asarray(reference_predictions), minlength=k)
        current_counts = np.bincount(np.asarray(current_predictions), minlength=k)
        chi2, chi2_p = class_distribution_drift(reference_counts, current_counts)
        tvd = total_variation_distance(reference_counts, current_counts)

    if current_confidence.size < 100:
        notes.append(
            f"NOTE: only {current_confidence.size} current samples — PSI is unstable on small "
            f"windows; treat this as indicative."
        )
    if current_confidence.mean() < reference_confidence.mean() - 0.05:
        notes.append(
            "NOTE: mean confidence has fallen materially, which is the classic signature of "
            "inputs moving away from the training distribution."
        )

    return DriftReport(
        n_reference=int(reference_confidence.size),
        n_current=int(current_confidence.size),
        confidence_psi=psi,
        confidence_ks_statistic=ks_statistic,
        confidence_ks_pvalue=ks_p,
        class_chi2=chi2,
        class_chi2_pvalue=chi2_p,
        class_tvd=tvd,
        mean_confidence_reference=float(reference_confidence.mean()),
        mean_confidence_current=float(current_confidence.mean()),
        notes=notes,
    )
