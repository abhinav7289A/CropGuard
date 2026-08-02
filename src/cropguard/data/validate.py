"""Data validation for the ImageFolder dataset.

Checks: image integrity (decodable), minimum resolution, minimum samples per class.
Writes a JSON report and exits non-zero on failure so it can gate the DVC pipeline / CI.

Usage:
    python -m cropguard.data.validate --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from cropguard.config import data_root, load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ValidationReport:
    total_images: int = 0
    corrupt: list[str] = field(default_factory=list)
    low_resolution: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    underpopulated_classes: dict[str, int] = field(default_factory=dict)
    # Classes with enough images overall but too few once the split takes its share. These
    # warn rather than fail - see thin_test_classes() for why.
    thin_test_classes: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.corrupt and not self.underpopulated_classes

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_images": self.total_images,
            "num_classes": len(self.class_counts),
            "corrupt_count": len(self.corrupt),
            "corrupt": self.corrupt[:50],
            "low_resolution_count": len(self.low_resolution),
            "low_resolution": self.low_resolution[:50],
            "class_counts": dict(sorted(self.class_counts.items())),
            "underpopulated_classes": self.underpopulated_classes,
            "thin_test_classes": self.thin_test_classes,
        }


def thin_test_classes(
    class_counts: dict[str, int], test_fraction: float, min_test_samples: int = 100
) -> dict[str, int]:
    """Classes whose *test* share is too small for per-class metrics to mean anything.

    The `min_samples_per_class` gate counts the whole dataset, which is the wrong quantity.
    Per-class recall is measured on the test split, and its standard error is
    `sqrt(p(1-p)/n_test)` - so 100 images total at a 15% test fraction leaves 15 test images
    and a confidence interval tens of points wide.

    Measured on this dataset: `Potato___healthy` has 152 images, clears the 100 gate, and ends
    up with 24 in test - recall 0.833 with a 95% interval of [0.641, 0.933]. Ten classes are
    in that position.

    This **warns rather than fails**. Refusing to train on an otherwise valid public dataset
    would be the wrong call; the honest response is to know which per-class numbers cannot
    carry weight, which is why `errors.py` reports Wilson intervals alongside them.
    """
    return {
        name: int(count * test_fraction)
        for name, count in sorted(class_counts.items())
        if int(count * test_fraction) < min_test_samples
    }


def validate_imagefolder(
    root: Path, min_resolution: int = 128, min_samples_per_class: int = 100
) -> ValidationReport:
    report = ValidationReport()
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found under {root}")

    for class_dir in class_dirs:
        count = 0
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            report.total_images += 1
            rel = str(img_path.relative_to(root))
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception:
                report.corrupt.append(rel)
                continue
            if min(width, height) < min_resolution:
                report.low_resolution.append(rel)
                continue
            count += 1
        report.class_counts[class_dir.name] = count
        if count < min_samples_per_class:
            report.underpopulated_classes[class_dir.name] = count
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=None,
        help="Override config (use a low value for smoke-test subsets)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["data"]["validation"]
    min_samples = args.min_samples_per_class or vcfg["min_samples_per_class"]
    root = data_root(cfg) / "plantvillage"

    report = validate_imagefolder(root, vcfg["min_resolution"], min_samples)
    report.thin_test_classes = thin_test_classes(
        report.class_counts,
        cfg["data"]["split"]["test"],
        args.min_test_samples_per_class,
    )
    out_path = root.parent / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Validated {report.total_images} images in {len(report.class_counts)} classes")
    print(
        f"Corrupt: {len(report.corrupt)} | Low-res: {len(report.low_resolution)} "
        f"| Underpopulated classes: {len(report.underpopulated_classes)}"
    )
    print(f"Report: {out_path}")

    if report.thin_test_classes:
        print()
        print(
            f"WARNING: {len(report.thin_test_classes)} classes will have fewer than "
            f"{args.min_test_samples_per_class} test images:"
        )
        for name, n in sorted(report.thin_test_classes.items(), key=lambda kv: kv[1])[:8]:
            print(f"  {name:52s} ~{n} in test")
        print("  Per-class metrics for these carry wide confidence intervals - see")
        print("  cropguard.evaluation.errors, which reports Wilson bounds alongside them.")

    if not report.passed:
        print("VALIDATION FAILED", file=sys.stderr)
        sys.exit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
