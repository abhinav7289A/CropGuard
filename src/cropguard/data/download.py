"""Download PlantVillage from HuggingFace and export to an ImageFolder layout.

The upstream repo (mohanty/PlantVillage) is a *script-based* dataset: a `plant_village.py`
loading script plus a single `data.zip`. `datasets` v4 dropped loading-script support, so
`load_dataset(..., "color")` now fails with `BuilderConfig 'color' not found`. We therefore
skip `datasets` entirely and read the archive directly — fewer moving parts, no re-encoding,
and it pins us to the exact bytes upstream published.

Archive layout:
    raw/<color|grayscale|segmented>/<class_name>/<filename>.jpg

Output layout (DVC-friendly, torchvision-friendly):
    <data_root>/plantvillage/<class_name>/<filename>.jpg
    <data_root>/plantvillage/manifest.json
    <data_root>/reference/{color_train.txt,color_test.txt,leaf-map.json}

Usage:
    python -m cropguard.data.download --config configs/base.yaml [--limit-per-class N]

`--limit-per-class` builds a small subset for local smoke tests.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from tqdm import tqdm

from cropguard.config import data_root, load_config

HF_DATASET_ID = "mohanty/PlantVillage"
HF_ARCHIVE = "data.zip"
IMAGE_VARIANT = "color"  # 'grayscale' and 'segmented' are the ablation variants
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Small auxiliary files worth keeping: the official train/test lists let us reproduce the
# published benchmark, and leaf-map.json groups images taken of the *same physical leaf* —
# needed for a leakage-free split (see fetch_reference_files).
REFERENCE_FILES = [
    f"splits/{IMAGE_VARIANT}_train.txt",
    f"splits/{IMAGE_VARIANT}_test.txt",
    "leaf_grouping/leaf-map.json",
]


def fetch_archive() -> Path:
    from huggingface_hub import hf_hub_download

    print(f"Fetching {HF_DATASET_ID}/{HF_ARCHIVE} (~2.2 GB, cached after the first run)...")
    return Path(hf_hub_download(HF_DATASET_ID, HF_ARCHIVE, repo_type="dataset"))


def fetch_reference_files(out_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_FILES:
        try:
            src = Path(hf_hub_download(HF_DATASET_ID, name, repo_type="dataset"))
            (out_dir / src.name).write_bytes(src.read_bytes())
        except Exception as exc:  # non-fatal: these are convenience artifacts
            print(f"  warning: could not fetch {name}: {exc}")


def export_imagefolder(
    archive_path: Path, out_dir: Path, limit_per_class: int | None = None
) -> dict:
    """Extract `raw/<variant>/<class>/*` from the archive into an ImageFolder tree."""
    prefix = f"raw/{IMAGE_VARIANT}/"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and info.filename.startswith(prefix)
            and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not members:
            raise RuntimeError(
                f"No images under {prefix!r} in {archive_path}. "
                f"Top-level entries: {sorted({n.split('/')[0] for n in archive.namelist()})}"
            )

        # Sort for a deterministic per-class ordering, which makes --limit-per-class
        # subsets reproducible across machines.
        members.sort(key=lambda info: info.filename)

        for info in tqdm(members, desc="Extracting images"):
            parts = info.filename.split("/")
            if len(parts) < 4:
                continue
            class_name, file_name = parts[2], parts[-1]

            seen = counts.get(class_name, 0)
            if limit_per_class is not None and seen >= limit_per_class:
                continue

            class_dir = out_dir / class_name
            class_dir.mkdir(exist_ok=True)
            with archive.open(info) as src:
                (class_dir / file_name).write_bytes(src.read())
            counts[class_name] = seen + 1

    manifest = {
        "source": f"{HF_DATASET_ID}:{HF_ARCHIVE}",
        "variant": IMAGE_VARIANT,
        "num_classes": len(counts),
        "num_images": sum(counts.values()),
        "class_counts": dict(sorted(counts.items())),
        "limit_per_class": limit_per_class,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Exported {manifest['num_images']} images / {manifest['num_classes']} classes -> {out_dir}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit-per-class", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = data_root(cfg)
    export_imagefolder(fetch_archive(), root / "plantvillage", args.limit_per_class)
    fetch_reference_files(root / "reference")


if __name__ == "__main__":
    main()
