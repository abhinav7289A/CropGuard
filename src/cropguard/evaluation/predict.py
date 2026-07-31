"""Run a model over a split and save per-image predictions for later statistical comparison.

Inference goes through the exported ONNX graph and the *serving* preprocessor, so the numbers
we run hypothesis tests on are the numbers production actually produces — not a torch-side
approximation of them. This also keeps evaluation torch-free.

Usage:
    python -m cropguard.evaluation.predict --model models/cropguard.onnx \
        --split test --out artifacts/preds_baseline.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cropguard.config import CONFIG_DIR, data_root, load_config
from cropguard.serving.model_loader import preprocess, softmax


def load_split(root: Path, split: str) -> list[str]:
    with open(root / "splits.json", encoding="utf-8") as f:
        return json.load(f)[split]


def predict_split(
    model_path: Path,
    root: Path,
    split: str,
    class_names: list[str],
    image_size: int = 224,
    batch_size: int = 32,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths = load_split(root, split)
    if limit is not None:
        paths = paths[:limit]
    image_root = root / "plantvillage"

    labels = np.array([class_to_idx[p.split("/")[0]] for p in paths], dtype=np.int64)
    probabilities = np.zeros((len(paths), len(class_names)), dtype=np.float32)

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        batch = np.concatenate(
            [preprocess((image_root / p).read_bytes(), image_size) for p in chunk], axis=0
        )
        logits = session.run(["logits"], {"input": batch})[0]
        probabilities[start : start + len(chunk)] = softmax(logits)
        if start % (batch_size * 20) == 0:
            print(f"  {min(start + len(chunk), len(paths))}/{len(paths)}", flush=True)

    predictions = probabilities.argmax(axis=1).astype(np.int64)
    return {
        "paths": np.array(paths),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "correct": (predictions == labels),
    }


def save_predictions(result: dict[str, np.ndarray], out_path: Path, **metadata) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **result, metadata=json.dumps(metadata))
    accuracy = float(result["correct"].mean())
    print(f"Saved {len(result['labels'])} predictions -> {out_path} (accuracy {accuracy:.4f})")


def load_predictions(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files if k != "metadata"}
        if "metadata" in data.files:
            result["metadata"] = json.loads(str(data["metadata"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", required=True, type=Path, help="ONNX model path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N images")
    parser.add_argument("--model-version", default="unknown")
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(CONFIG_DIR / "classes.json", encoding="utf-8") as f:
        class_names = json.load(f)

    root = data_root(cfg)
    result = predict_split(
        args.model,
        root,
        args.split,
        class_names,
        image_size=cfg["data"]["image_size"],
        batch_size=args.batch_size,
        limit=args.limit,
    )
    save_predictions(
        result,
        args.out,
        model=str(args.model),
        model_version=args.model_version,
        split=args.split,
        num_classes=len(class_names),
    )


if __name__ == "__main__":
    main()
