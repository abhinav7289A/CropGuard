"""Error analysis: which classes fail, which get confused, and which failures matter.

A single accuracy figure hides everything actionable. This module answers three questions:

1. **Which classes are weak, and is that signal or noise?** Per-class recall is a binomial
   proportion, and eight of our classes have fewer than 100 test images. `Potato___healthy`
   shows recall 0.833 — which is four mistakes out of 24. Every per-class figure here carries
   a Wilson interval so a small-sample class cannot be mistaken for a real weakness.

2. **Which classes get confused with which?** Off-diagonal mass, ranked. For this model it
   concentrates in two biologically plausible pairs, which is a far better answer to "what
   does it get wrong" than a number.

3. **Which errors are dangerous?** A wrong prediction made at high confidence is worse than a
   wrong prediction the model was unsure about — it is the kind a user acts on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """95% CI for a binomial proportion, Wilson score method.

    Not the normal approximation `p ± z·sqrt(p(1-p)/n)`: that degenerates badly at small n or
    extreme p — at p=1.0 it returns a zero-width interval, claiming certainty from a handful
    of samples. Wilson stays inside [0, 1] and keeps sensible width. With eight classes under
    100 test images, this is the difference between an honest table and a misleading one.
    """
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z / denominator * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


@dataclass
class ClassMetrics:
    name: str
    support: int
    precision: float
    recall: float
    f1: float
    recall_ci: tuple[float, float]

    @property
    def is_reliable(self) -> bool:
        """Enough support for the per-class number to be worth acting on."""
        return self.support >= 100

    @property
    def ci_width(self) -> float:
        return self.recall_ci[1] - self.recall_ci[0]

    def summary(self) -> str:
        flag = "" if self.is_reliable else "  (low support - treat as indicative)"
        return (
            f"{self.name:52s} n={self.support:5d}  P={self.precision:.3f}  "
            f"R={self.recall:.3f} [{self.recall_ci[0]:.3f}, {self.recall_ci[1]:.3f}]  "
            f"F1={self.f1:.3f}{flag}"
        )


def per_class_metrics(
    labels: np.ndarray, predictions: np.ndarray, class_names: list[str]
) -> list[ClassMetrics]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    out: list[ClassMetrics] = []
    for index, name in enumerate(class_names):
        actual = labels == index
        predicted = predictions == index
        true_positive = int((actual & predicted).sum())
        support = int(actual.sum())

        precision = true_positive / max(int(predicted.sum()), 1)
        recall = true_positive / max(support, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        out.append(
            ClassMetrics(
                name=name,
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
                recall_ci=wilson_interval(true_positive, support),
            )
        )
    return out


@dataclass
class ConfusionPair:
    true_class: str
    predicted_class: str
    count: int
    rate: float  # share of the true class misrouted this way

    def summary(self) -> str:
        return (
            f"{self.true_class:45s} -> {self.predicted_class:45s} "
            f"{self.count:4d} ({self.rate:.1%} of class)"
        )


def confusion_pairs(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    top_n: int = 10,
    min_count: int = 2,
) -> list[ConfusionPair]:
    """Most-confused (true, predicted) pairs, ranked by count.

    Ranked by absolute count rather than rate: a 50% error rate on a 4-image class is noise,
    while 30 misroutes out of 300 is a pattern worth investigating. `rate` is reported
    alongside so both readings are available.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    pairs: list[ConfusionPair] = []
    wrong = labels != predictions
    for true_index in np.unique(labels[wrong]):
        mask = wrong & (labels == true_index)
        support = int((labels == true_index).sum())
        for predicted_index, count in zip(
            *np.unique(predictions[mask], return_counts=True), strict=True
        ):
            if count < min_count:
                continue
            pairs.append(
                ConfusionPair(
                    true_class=class_names[int(true_index)],
                    predicted_class=class_names[int(predicted_index)],
                    count=int(count),
                    rate=int(count) / max(support, 1),
                )
            )
    pairs.sort(key=lambda p: p.count, reverse=True)
    return pairs[:top_n]


def confident_mistakes(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    paths: np.ndarray | None = None,
    top_n: int = 20,
) -> list[dict]:
    """Wrong predictions ranked by the confidence with which they were made.

    These are the errors that matter operationally: a user acts on a confident answer. A model
    that is unsure and wrong is behaving reasonably; one that is certain and wrong is not.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    wrong = np.flatnonzero(predictions != labels)
    ranked = wrong[np.argsort(confidence[wrong])[::-1]][:top_n]

    return [
        {
            "path": str(paths[i]) if paths is not None else None,
            "true_class": class_names[int(labels[i])],
            "predicted_class": class_names[int(predictions[i])],
            "confidence": float(confidence[i]),
            "true_class_probability": float(probabilities[i, labels[i]]),
        }
        for i in ranked
    ]


def error_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    paths: np.ndarray | None = None,
) -> dict:
    """Everything above in one call, shaped for a report."""
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    predictions = probabilities.argmax(axis=1)

    metrics = per_class_metrics(labels, predictions, class_names)
    reliable = [m for m in metrics if m.is_reliable]

    return {
        "n_images": int(len(labels)),
        "n_errors": int((predictions != labels).sum()),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(np.mean([m.f1 for m in metrics])),
        "low_support_classes": [m.name for m in metrics if not m.is_reliable],
        "weakest_reliable_classes": [m.name for m in sorted(reliable, key=lambda m: m.f1)[:5]],
        "confusion_pairs": [p.__dict__ for p in confusion_pairs(labels, predictions, class_names)],
        "confident_mistakes": confident_mistakes(probabilities, labels, class_names, paths),
    }
