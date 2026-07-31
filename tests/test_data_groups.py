from __future__ import annotations

from collections import Counter, defaultdict

from cropguard.data.groups import assign_groups, is_resolved, leaf_id, leakage_report
from cropguard.data.split import grouped_split

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

LEAF_MAP = {
    "0001": ["Tomato___healthy:::11.0"],
    "0002": ["Tomato___healthy:::11.0"],
    # Ambiguous identifier: resolution must pick the suggestion matching the image's class.
    "0003": ["Potato___healthy:::9.0", "Tomato___healthy:::12.0"],
}


def test_leaf_id_groups_images_of_the_same_leaf():
    assert leaf_id("Tomato___healthy/0001.jpg", LEAF_MAP) == "Tomato___healthy:::11.0"
    assert leaf_id("Tomato___healthy/0002.jpg", LEAF_MAP) == "Tomato___healthy:::11.0"


def test_leaf_id_disambiguates_by_class():
    assert leaf_id("Tomato___healthy/0003.jpg", LEAF_MAP) == "Tomato___healthy:::12.0"
    assert leaf_id("Potato___healthy/0003.jpg", LEAF_MAP) == "Potato___healthy:::9.0"


def test_leaf_id_strips_upstream_filename_decorations():
    """The derivation must mirror the upstream loading script's identifier normalization."""
    assert leaf_id("Tomato___healthy/0001_final_masked.jpg", LEAF_MAP) == "Tomato___healthy:::11.0"
    assert leaf_id("Tomato___healthy/xyz___0001.JPG", LEAF_MAP) == "Tomato___healthy:::11.0"
    assert leaf_id("Tomato___healthy/0001copy2.jpg", LEAF_MAP) == "Tomato___healthy:::11.0"


def test_unresolved_images_fall_back_to_their_own_group():
    first = leaf_id("Tomato___healthy/9999.jpg", LEAF_MAP)
    assert not is_resolved(first)
    # Fallbacks are class-scoped, so an identical filename in another class cannot merge in.
    assert first != leaf_id("Potato___healthy/9999.jpg", LEAF_MAP)


def test_assign_groups_is_positional():
    paths = ["Tomato___healthy/0001.jpg", "Tomato___healthy/9999.jpg"]
    groups = assign_groups(paths, LEAF_MAP)
    assert len(groups) == 2
    assert is_resolved(groups[0]) and not is_resolved(groups[1])


# --- grouped_split ------------------------------------------------------------------


def _synthetic(num_classes: int = 4, groups_per_class: int = 40, images_per_group: int = 5):
    """Paths/labels/groups where every leaf contributes several near-duplicate images."""
    paths, labels, groups = [], [], []
    for label in range(num_classes):
        for g in range(groups_per_class):
            for i in range(images_per_group):
                paths.append(f"class{label}/g{g:03d}_{i}.jpg")
                labels.append(label)
                groups.append(f"class{label}:::leaf{g:03d}")
    return paths, labels, groups


def test_no_group_is_split_across_partitions():
    paths, labels, groups = _synthetic()
    group_of = dict(zip(paths, groups, strict=True))
    splits = grouped_split(paths, labels, groups, FRACTIONS, seed=42)

    where = defaultdict(set)
    for name, split_paths in splits.items():
        for path in split_paths:
            where[group_of[path]].add(name)
    assert all(len(names) == 1 for names in where.values())


def test_split_is_exhaustive_and_disjoint():
    paths, labels, groups = _synthetic()
    splits = grouped_split(paths, labels, groups, FRACTIONS, seed=42)
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))

    assert train | val | test == set(paths)
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) + len(val) + len(test) == len(paths)


def test_every_class_stays_close_to_the_target_fractions():
    paths, labels, groups = _synthetic()
    splits = grouped_split(paths, labels, groups, FRACTIONS, seed=42)
    counts = {k: Counter(p.split("/")[0] for p in v) for k, v in splits.items()}

    for class_name in {p.split("/")[0] for p in paths}:
        total = sum(counts[s][class_name] for s in counts)
        for split_name, target in FRACTIONS.items():
            actual = counts[split_name][class_name] / total
            assert abs(actual - target) < 0.05, f"{class_name}/{split_name}: {actual:.3f}"


def test_grouped_split_is_deterministic_for_a_given_seed():
    paths, labels, groups = _synthetic()
    assert grouped_split(paths, labels, groups, FRACTIONS, seed=42) == grouped_split(
        paths, labels, groups, FRACTIONS, seed=42
    )


def test_uneven_group_sizes_still_pack_close_to_target():
    """Real leaves range from 1 to 33 images; the packer must cope with that skew."""
    paths, labels, groups = [], [], []
    for g in range(60):
        for i in range(1 + (g % 12)):  # group sizes 1..12
            paths.append(f"class0/g{g:03d}_{i}.jpg")
            labels.append(0)
            groups.append(f"class0:::leaf{g:03d}")

    splits = grouped_split(paths, labels, groups, FRACTIONS, seed=0)
    for split_name, target in FRACTIONS.items():
        assert abs(len(splits[split_name]) / len(paths) - target) < 0.05


def test_leakage_report_counts_only_resolved_groups():
    splits = {
        "train": ["c/a.jpg", "c/b.jpg"],
        "val": ["c/c.jpg"],
        "test": ["c/d.jpg", "c/e.jpg"],
    }
    groups = {
        "c/a.jpg": "leaf1",
        "c/b.jpg": "leaf2",
        "c/c.jpg": "leaf1",  # leaked into val
        "c/d.jpg": "leaf3",  # clean
        "c/e.jpg": "fallback_c:::e",  # unresolved -> excluded from the ratio
    }
    report = leakage_report(splits, groups)

    assert report["val"]["leaked"] == 1
    assert report["val"]["leaked_pct_of_resolved"] == 100.0
    assert report["test"]["leaked"] == 0
    assert report["test"]["resolved"] == 1  # the fallback image is not counted


def test_grouped_split_removes_the_leakage_a_naive_split_creates():
    """End-to-end guard: this is the property the whole module exists to provide."""
    paths, labels, groups = _synthetic()
    group_of = dict(zip(paths, groups, strict=True))
    splits = grouped_split(paths, labels, groups, FRACTIONS, seed=42)

    report = leakage_report(splits, group_of)
    assert report["val"]["leaked"] == 0
    assert report["test"]["leaked"] == 0
