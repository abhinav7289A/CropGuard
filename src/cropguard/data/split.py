"""Train/val/test split with persisted indices for reproducibility.

Two strategies:
  grouped     (default) — stratified by class AND grouped by leaf, so every image of a given
                          physical leaf lands in exactly one split. This is the honest split.
  stratified            — the naive per-image stratified split. Kept only so the leakage it
                          causes can be quantified and reported, never for headline metrics.

Writes:
    <data_root>/splits.json       — {"train": [...], "val": [...], "test": [...]} of rel paths
    <data_root>/split_report.json — strategy, per-split sizes, and the leakage measurement
    configs/classes.json          — ordered class names (the single source of truth for label
                                    order, shared by training, ONNX export, and the API)

Usage:
    python -m cropguard.data.split --config configs/base.yaml
    python -m cropguard.data.split --config configs/base.yaml --strategy stratified
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from cropguard.config import CONFIG_DIR, data_root, load_config
from cropguard.data.groups import assign_groups, leakage_report, load_leaf_map
from cropguard.data.validate import IMAGE_EXTENSIONS

SPLIT_NAMES = ("train", "val", "test")


def collect_samples(root: Path) -> tuple[list[str], list[int], list[str]]:
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    paths: list[str] = []
    labels: list[int] = []
    for idx, name in enumerate(class_names):
        for img_path in sorted((root / name).iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(img_path.relative_to(root)).replace("\\", "/"))
                labels.append(idx)
    return paths, labels, class_names


def stratified_split(
    paths: list[str], labels: list[int], fractions: dict[str, float], seed: int
) -> dict[str, list[str]]:
    test_frac = fractions["test"]
    val_frac = fractions["val"]
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, test_size=test_frac + val_frac, stratify=labels, random_state=seed
    )
    val_paths, test_paths = train_test_split(
        rest_paths,
        test_size=test_frac / (test_frac + val_frac),
        stratify=rest_labels,
        random_state=seed,
    )
    return {"train": train_paths, "val": val_paths, "test": test_paths}


def grouped_split(
    paths: list[str],
    labels: list[int],
    groups: list[str],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Stratified by class, grouped by leaf — whole groups are assigned, never split.

    Groups never span classes in PlantVillage (verified: 0 of 20,015), so each class is
    packed independently. Within a class we walk its groups largest-first and hand each to
    whichever split is furthest below its target count — the classic longest-processing-time
    heuristic, which keeps every class close to the requested fractions despite group sizes
    varying from 1 to 33 images.
    """
    by_class: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path, label, group in zip(paths, labels, groups, strict=True):
        by_class[label][group].append(path)

    rng = random.Random(seed)
    result: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    for label in sorted(by_class):
        class_groups = by_class[label]
        total = sum(len(members) for members in class_groups.values())

        # Shuffle first so that equal-sized groups are ordered reproducibly but not by name,
        # then sort largest-first for the packing heuristic.
        names = sorted(class_groups)
        rng.shuffle(names)
        names.sort(key=lambda g: len(class_groups[g]), reverse=True)

        assigned = dict.fromkeys(SPLIT_NAMES, 0)
        for group in names:
            target = max(SPLIT_NAMES, key=lambda s: fractions[s] * total - assigned[s])
            result[target].extend(class_groups[group])
            assigned[target] += len(class_groups[group])

    for name in SPLIT_NAMES:
        result[name].sort()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--strategy", choices=["grouped", "stratified"], default="grouped")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = data_root(cfg) / "plantvillage"
    paths, labels, class_names = collect_samples(root)
    fractions = cfg["data"]["split"]

    leaf_map_path = root.parent / "reference" / "leaf-map.json"
    groups_by_path: dict[str, str] | None = None

    if args.strategy == "grouped":
        if not leaf_map_path.exists():
            raise FileNotFoundError(
                f"{leaf_map_path} not found — re-run cropguard.data.download to fetch it, "
                f"or pass --strategy stratified (which leaks leaves across splits)."
            )
        groups = assign_groups(paths, load_leaf_map(leaf_map_path))
        groups_by_path = dict(zip(paths, groups, strict=True))
        splits = grouped_split(paths, labels, groups, fractions, cfg["seed"])
    else:
        splits = stratified_split(paths, labels, fractions, cfg["seed"])
        if leaf_map_path.exists():
            groups_by_path = dict(
                zip(paths, assign_groups(paths, load_leaf_map(leaf_map_path)), strict=True)
            )

    splits_path = root.parent / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    classes_path = CONFIG_DIR / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    sizes = {k: len(v) for k, v in splits.items()}
    report = {
        "strategy": args.strategy,
        "seed": cfg["seed"],
        "fractions": fractions,
        "sizes": sizes,
        "num_classes": len(class_names),
        "num_images": len(paths),
    }
    if groups_by_path is not None:
        report["leakage"] = leakage_report(splits, groups_by_path)

    with open(root.parent / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Split {len(paths)} images -> {sizes} (strategy={args.strategy}, seed={cfg['seed']})")
    if groups_by_path is not None:
        for name in ("val", "test"):
            stats = report["leakage"][name]
            print(
                f"  {name}: {stats['leaked']}/{stats['resolved']} resolved images share a leaf "
                f"with train ({stats['leaked_pct_of_resolved']}%)"
            )
    print(f"Wrote {splits_path} and {classes_path} ({len(class_names)} classes)")


if __name__ == "__main__":
    main()
