"""Confidence calibration: is a prediction reported at 0.9 correct 90% of the time?

Accuracy says nothing about this. A model can be 99% accurate and badly calibrated — and this
one is, by construction. Label smoothing (eps=0.1, K=38) bounds the achievable softmax output
at `(1 - eps) + eps/K = 0.9026`, and live predictions sit exactly there. Confidence is
therefore systematically *under*stated: the model is far more certain than it can report.

That matters the moment a number is shown to a user, or used as a threshold for "refer this to
an agronomist".

Temperature scaling (Guo et al., 2017) is the standard fix: divide logits by a single learned
scalar T before the softmax.

    p_i = softmax(z_i / T)

- T > 1 softens the distribution (fixes over-confidence)
- T < 1 sharpens it (fixes under-confidence — what we expect here)

One parameter, fitted by minimising NLL on held-out data. Because it is monotonic it **cannot
change any prediction**, only the probabilities attached to them — accuracy is provably
unchanged, which is what makes it safe to apply post-hoc.

**T is fitted on validation, never test.** Fitting on test would tune the calibration to the
set used to report calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with optional temperature. Max-subtracted for overflow safety."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    accuracy: float
    mean_confidence: float
    n_bins: int

    @property
    def gap(self) -> float:
        """Mean confidence minus accuracy. Positive = over-confident, negative = under."""
        return self.mean_confidence - self.accuracy

    def summary(self) -> str:
        direction = "over-confident" if self.gap > 0 else "under-confident"
        return (
            f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  Brier={self.brier:.4f} | "
            f"acc={self.accuracy:.4f} conf={self.mean_confidence:.4f} "
            f"(gap {self.gap:+.4f}, {direction})"
        )


def _binned(confidence: np.ndarray, correct: np.ndarray, n_bins: int):
    """Yield (bin_weight, |accuracy - confidence|) for each non-empty equal-width bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        # Left-open intervals so confidence == 1.0 lands in the final bin rather than nowhere.
        mask = (confidence > low) & (confidence <= high) if low > 0 else (confidence <= high)
        if not mask.any():
            continue
        yield mask.mean(), abs(correct[mask].mean() - confidence[mask].mean())


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Weighted mean gap between confidence and accuracy across confidence bins.

        ECE = sum_b (n_b / n) * |acc(b) - conf(b)|

    Weighted by bin population, so bins holding few predictions cannot dominate. 0 is perfect.
    """
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(labels)
    return float(sum(w * gap for w, gap in _binned(confidence, correct, n_bins)))


def maximum_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Worst gap over any bin — the tail risk ECE averages away."""
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(labels)
    gaps = [gap for _, gap in _binned(confidence, correct, n_bins)]
    return float(max(gaps)) if gaps else 0.0


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target.

    Unlike ECE it is a *proper scoring rule* — it cannot be gamed by a model that reports a
    constant confidence equal to its accuracy, which would score a perfect ECE. Lower is
    better.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(labels)), np.asarray(labels)] = 1.0
    return float(((probabilities - onehot) ** 2).sum(axis=1).mean())


def evaluate_calibration(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> CalibrationReport:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    correct = probabilities.argmax(axis=1) == labels
    return CalibrationReport(
        ece=expected_calibration_error(probabilities, labels, n_bins),
        mce=maximum_calibration_error(probabilities, labels, n_bins),
        brier=brier_score(probabilities, labels),
        accuracy=float(correct.mean()),
        mean_confidence=float(probabilities.max(axis=1).mean()),
        n_bins=n_bins,
    )


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, np.ndarray]:
    """Per-bin confidence, accuracy and count — the data behind a reliability diagram.

    A perfectly calibrated model traces the diagonal; points below it are over-confident.
    """
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(labels)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, counts, centers = [], [], [], []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high) if low > 0 else (confidence <= high)
        centers.append((low + high) / 2)
        if mask.any():
            conf.append(confidence[mask].mean())
            acc.append(correct[mask].mean())
            counts.append(int(mask.sum()))
        else:
            conf.append(np.nan)
            acc.append(np.nan)
            counts.append(0)
    return {
        "bin_center": np.array(centers),
        "confidence": np.array(conf),
        "accuracy": np.array(acc),
        "count": np.array(counts),
    }


class TemperatureScaler:
    """Learns a single scalar T minimising NLL on held-out logits.

    Deliberately one parameter: with 38 classes, anything richer (per-class temperature,
    vector scaling) has enough freedom to overfit the calibration set, which is usually far
    smaller than the training set.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.fitted: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> TemperatureScaler:
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels)
        rows = np.arange(len(labels))

        def nll(log_t: np.ndarray) -> float:
            # Optimise in log-space so T stays strictly positive without a constrained solver.
            probabilities = softmax(logits, float(np.exp(log_t[0])))
            return float(-np.log(probabilities[rows, labels] + 1e-12).mean())

        result = optimize.minimize(nll, x0=np.array([0.0]), method="Nelder-Mead")
        self.temperature = float(np.exp(result.x[0]))
        self.fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureScaler must be fitted before transform()")
        return softmax(logits, self.temperature)

    def __repr__(self) -> str:
        state = f"T={self.temperature:.4f}" if self.fitted else "unfitted"
        return f"TemperatureScaler({state})"
