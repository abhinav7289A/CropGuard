"""A/B compare two saved prediction sets and report whether the difference is real.

Usage:
    python -m cropguard.evaluation.compare \
        --baseline artifacts/preds_baseline.npz \
        --challenger artifacts/preds_challenger.npz \
        --out artifacts/comparison.json

Exit code is 0 when the challenger wins on the paired tests and 1 otherwise, so CI can gate
model promotion on statistical evidence rather than a raw accuracy delta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cropguard.evaluation.hypothesis import compare_models
from cropguard.evaluation.predict import load_predictions


def _assert_same_holdout(baseline: dict, challenger: dict) -> None:
    """The tests are paired — comparing predictions over different images is meaningless."""
    if baseline["labels"].shape != challenger["labels"].shape:
        raise ValueError(
            f"Different holdout sizes: {baseline['labels'].shape} vs {challenger['labels'].shape}"
        )
    if "paths" in baseline and "paths" in challenger:
        if not np.array_equal(baseline["paths"], challenger["paths"]):
            raise ValueError("Prediction files cover different images (or a different order)")
    if not np.array_equal(baseline["labels"], challenger["labels"]):
        raise ValueError("Ground-truth labels disagree between the two prediction files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--challenger", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline = load_predictions(args.baseline)
    challenger = load_predictions(args.challenger)
    _assert_same_holdout(baseline, challenger)

    result = compare_models(
        baseline["correct"],
        challenger["correct"],
        labels=baseline["labels"],
        n_resamples=args.resamples,
        seed=args.seed,
    )

    print(f"baseline   : {args.baseline}")
    print(f"challenger : {args.challenger}")
    print()
    print(result.summary())

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload["baseline_file"] = str(args.baseline)
        payload["challenger_file"] = str(args.challenger)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nReport: {args.out}")

    sys.exit(0 if result.challenger_is_better else 1)


if __name__ == "__main__":
    main()
