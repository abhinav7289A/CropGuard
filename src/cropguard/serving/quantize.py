"""Static INT8 quantization of the exported ONNX graph.

Why static rather than dynamic: `quantize_dynamic` rewrites every `Conv` into `ConvInteger`,
which ONNX Runtime's CPU backend has no optimized kernel for. Measured on this model it ran at
1567 ms/image against 19 ms for fp32 - a 75x regression bought with a 4x size reduction.
Dynamic quantization targets MatMul-dominated architectures (Transformers, RNNs); a
Conv-dominated CNN needs static quantization, which emits `QLinearConv`.

The price is a calibration pass: activation ranges are measured up front on real images
instead of being recomputed per inference.

**Calibration uses the validation split, never test.** The test set decides whether the model
ships; letting it influence how the model is built would contaminate that decision, in exactly
the way that makes a holdout worthless.

Usage:
    python -m cropguard.serving.quantize --model models/cropguard.onnx \
        --out models/cropguard.int8.onnx --num-samples 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cropguard.serving.model_loader import preprocess


class ImageCalibrationReader:
    """Feeds real preprocessed images to the calibrator, one batch at a time.

    Calibration must see the *serving* preprocessing, not an approximation of it: the
    activation ranges recorded here become fixed scale factors at inference, so a mismatch
    bakes a systematic error into the quantized graph.
    """

    def __init__(
        self,
        paths: list[str],
        image_root: Path,
        image_size: int = 224,
        batch_size: int = 8,
        input_name: str = "input",
    ) -> None:
        self.paths = paths
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.input_name = input_name
        self.rewind()

    def rewind(self) -> None:
        self._index = 0

    def get_next(self) -> dict | None:
        if self._index >= len(self.paths):
            return None
        chunk = self.paths[self._index : self._index + self.batch_size]
        self._index += self.batch_size
        batch = np.concatenate(
            [preprocess((self.image_root / p).read_bytes(), self.image_size) for p in chunk],
            axis=0,
        )
        return {self.input_name: batch}


def calibration_paths(data_root: Path, num_samples: int, seed: int = 42) -> list[str]:
    """A class-spread sample of the validation split.

    Sampling evenly across classes rather than at random matters here: activation ranges are
    driven by the extremes a layer sees, and a random draw from a 36x-imbalanced split would
    barely touch the rare classes.
    """
    with open(Path(data_root) / "splits.json", encoding="utf-8") as f:
        val_paths = json.load(f)["val"]

    by_class: dict[str, list[str]] = {}
    for path in val_paths:
        by_class.setdefault(path.split("/")[0], []).append(path)

    rng = np.random.default_rng(seed)
    per_class = max(1, num_samples // len(by_class))
    sampled: list[str] = []
    for class_name in sorted(by_class):
        members = by_class[class_name]
        take = min(per_class, len(members))
        sampled.extend(rng.choice(members, size=take, replace=False).tolist())
    return sampled[:num_samples] if len(sampled) > num_samples else sampled


def quantize(
    model_path: Path,
    out_path: Path,
    data_root: Path,
    num_samples: int = 256,
    image_size: int = 224,
    per_channel: bool = True,
) -> dict:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    paths = calibration_paths(data_root, num_samples)
    print(f"Calibrating on {len(paths)} validation images...")

    # Same pre-processing pass the dynamic path needed: constant folding leaves conv weights
    # as Constant nodes, and the quantizer only rewrites initializers.
    prepared = out_path.with_suffix(".prepared.onnx")
    quant_pre_process(str(model_path), str(prepared), skip_symbolic_shape=False)

    reader = ImageCalibrationReader(paths, Path(data_root) / "plantvillage", image_size)

    quantize_static(
        model_input=str(prepared),
        model_output=str(out_path),
        calibration_data_reader=reader,
        # QOperator emits QLinearConv directly. QDQ is more portable but relies on the runtime
        # fusing Q/DQ pairs back into QLinearConv, which is the step that failed us before -
        # so prefer the form that does not depend on an optimization pass firing.
        quant_format=QuantFormat.QOperator,
        # Per-channel weight scales: a single scale per tensor is set by the widest channel,
        # crushing the resolution of narrow ones. Cheap accuracy, no runtime cost.
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prepared.unlink(missing_ok=True)

    stats = {
        "fp32_mb": round(model_path.stat().st_size / 1e6, 1),
        "int8_mb": round(out_path.stat().st_size / 1e6, 1),
        "calibration_images": len(paths),
    }
    stats["compression"] = round(stats["fp32_mb"] / stats["int8_mb"], 2)
    print(
        f"Quantized -> {out_path} "
        f"({stats['int8_mb']} MB vs {stats['fp32_mb']} MB, {stats['compression']}x)"
    )
    return stats


def summarize_ops(model_path: Path) -> dict[str, int]:
    from collections import Counter

    import onnx

    model = onnx.load(str(model_path))
    return dict(Counter(node.op_type for node in model.graph.node).most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default=Path("models/cropguard.onnx"), type=Path)
    parser.add_argument("--out", default=Path("models/cropguard.int8.onnx"), type=Path)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--no-per-channel", action="store_true")
    args = parser.parse_args()

    from cropguard.config import data_root, load_config

    cfg = load_config(args.config)
    stats = quantize(
        args.model,
        args.out,
        data_root(cfg),
        num_samples=args.num_samples,
        image_size=cfg["data"]["image_size"],
        per_channel=not args.no_per_channel,
    )

    ops = summarize_ops(args.out)
    print("\nquantized graph ops:", dict(list(ops.items())[:6]))
    if "ConvInteger" in ops:
        raise SystemExit(
            "ConvInteger in the quantized graph - ONNX Runtime has no optimized CPU kernel "
            "for it and inference will be far slower than fp32. Expected QLinearConv."
        )
    print(f"QLinearConv nodes: {ops.get('QLinearConv', 0)}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
