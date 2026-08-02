"""Fit temperature scaling on validation and report calibration on test.

    python -m cropguard.evaluation.calibrate \
        --val artifacts/preds_val.npz --test artifacts/preds_fp32.npz \
        --out artifacts/calibration.json

**The fit uses validation only.** Fitting T on the test set would tune the calibration to the
very data used to report it, which is the same mistake as tuning hyperparameters on test.

Temperature scaling cannot change a prediction — dividing logits by a positive scalar is
monotonic — so accuracy is provably identical before and after. The CLI asserts that rather
than trusting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cropguard.evaluation.calibration import (
    TemperatureScaler,
    evaluate_calibration,
    reliability_curve,
)


def logits_from(data: dict) -> np.ndarray:
    """Prefer stored logits; fall back to log(probabilities).

    The fallback is exact, not an approximation: log(p) = z - logsumexp(z), and softmax is
    shift-invariant, so softmax(log(p)/T) == softmax(z/T). It exists for prediction files
    written before logits were stored.
    """
    if "logits" in data:
        return np.asarray(data["logits"], dtype=np.float64)
    return np.log(np.clip(np.asarray(data["probabilities"], dtype=np.float64), 1e-12, None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", required=True, type=Path, help="validation predictions (.npz)")
    parser.add_argument("--test", required=True, type=Path, help="test predictions (.npz)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    val = dict(np.load(args.val, allow_pickle=False))
    test = dict(np.load(args.test, allow_pickle=False))

    scaler = TemperatureScaler().fit(logits_from(val), val["labels"])

    before = evaluate_calibration(test["probabilities"], test["labels"], args.bins)
    calibrated = scaler.transform(logits_from(test))
    after = evaluate_calibration(calibrated, test["labels"], args.bins)

    assert before.accuracy == after.accuracy, (
        "temperature scaling changed a prediction — it is monotonic and must not"
    )

    direction = (
        "sharpening (model was under-confident)"
        if scaler.temperature < 1
        else ("softening (model was over-confident)")
    )
    print(f"T = {scaler.temperature:.4f}  -> {direction}")
    print(f"fitted on {len(val['labels'])} validation images")
    print()
    print(f"{'':8s} {'ECE':>8s} {'MCE':>8s} {'Brier':>8s} {'conf':>8s} {'acc':>8s}")
    print(
        f"{'before':8s} {before.ece:8.4f} {before.mce:8.4f} {before.brier:8.4f} "
        f"{before.mean_confidence:8.4f} {before.accuracy:8.4f}"
    )
    print(
        f"{'after':8s} {after.ece:8.4f} {after.mce:8.4f} {after.brier:8.4f} "
        f"{after.mean_confidence:8.4f} {after.accuracy:8.4f}"
    )
    print()
    print(f"ECE   {before.ece:.4f} -> {after.ece:.4f}")
    print(f"Brier {before.brier:.4f} -> {after.brier:.4f}")
    if after.mce > before.mce:
        print(
            "\nNote: MCE rose. It is the worst *single* bin, and calibration concentrates "
            "most predictions into one high-confidence bin, leaving sparse bins where a "
            "handful of errors makes a large gap. ECE is population-weighted; MCE is not."
        )

    if args.out is not None:
        curve = reliability_curve(calibrated, test["labels"], args.bins)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "temperature": scaler.temperature,
                    "fitted_on": str(args.val),
                    "n_fit": int(len(val["labels"])),
                    "evaluated_on": str(args.test),
                    "n_eval": int(len(test["labels"])),
                    "before": before.__dict__,
                    "after": after.__dict__,
                    "reliability_after": {k: v.tolist() for k, v in curve.items()},
                },
                f,
                indent=2,
            )
        print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
