"""Leaf grouping — the basis for a leakage-free split.

PlantVillage photographed each physical leaf ~7 times from different angles (54,305 images
but only ~7,600 resolvable leaves). A plain stratified split scatters those near-duplicates
across train and test, so a model can memorize a leaf and "recognize" it again at test time.
Measured on the naive split: 74% of test images shared a leaf with train.

`leaf-map.json` (shipped in the upstream repo, saved to <data_root>/reference/ by
cropguard.data.download) maps a filename-derived identifier to leaf IDs. The derivation
below mirrors the upstream loading script so our groups match the published ones.
"""

from __future__ import annotations

import json
from pathlib import Path

FALLBACK_PREFIX = "fallback"
_STRIPPED_EXTENSIONS = (".jpg", ".JPG", ".png", ".PNG")


def load_leaf_map(path: str | Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def leaf_id(rel_path: str, leaf_map: dict[str, list[str]]) -> str:
    """Group key for one image, given its `<class_name>/<file_name>` relative path.

    Images the map cannot resolve (~24%) fall back to a class-scoped filename identifier.
    That is deliberately conservative: an unresolved image forms its own tiny group rather
    than being silently merged into someone else's leaf.
    """
    parts = rel_path.split("/")
    class_name, file_name = parts[0], parts[-1]

    identifier = file_name.replace("_final_masked", "")
    if "___" in identifier:
        identifier = identifier.split("___")[-1]
    identifier = identifier.split("copy")[0]
    for extension in _STRIPPED_EXTENSIONS:
        identifier = identifier.replace(extension, "")
    identifier = identifier.strip()

    suggestions = leaf_map.get(identifier.lower())
    if suggestions:
        if len(suggestions) == 1:
            return suggestions[0]
        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion
    return f"{FALLBACK_PREFIX}_{class_name}:::{identifier}"


def assign_groups(paths: list[str], leaf_map: dict[str, list[str]]) -> list[str]:
    return [leaf_id(path, leaf_map) for path in paths]


def is_resolved(group: str) -> bool:
    """True when the group came from leaf-map.json rather than the filename fallback."""
    return not group.startswith(f"{FALLBACK_PREFIX}_")


def leakage_report(splits: dict[str, list[str]], groups: dict[str, str]) -> dict:
    """Fraction of val/test images sharing a leaf group with train.

    Only resolved groups count: fallback groups are per-image by construction, so counting
    them would flatter the number rather than measure anything.
    """
    train_groups = {groups[p] for p in splits["train"] if is_resolved(groups[p])}
    report: dict = {"train_groups": len(train_groups)}
    for name in ("val", "test"):
        paths = splits[name]
        resolved = [p for p in paths if is_resolved(groups[p])]
        leaked = [p for p in resolved if groups[p] in train_groups]
        report[name] = {
            "images": len(paths),
            "resolved": len(resolved),
            "leaked": len(leaked),
            "leaked_pct_of_split": round(100 * len(leaked) / max(len(paths), 1), 2),
            "leaked_pct_of_resolved": round(100 * len(leaked) / max(len(resolved), 1), 2),
        }
    return report
