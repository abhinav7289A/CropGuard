"""ONNX Runtime inference wrapper — no torch dependency."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bytes: bytes, image_size: int = 224) -> np.ndarray:
    """Mirror eval_transforms: resize shorter side to size*256/224, center crop, normalize."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        image = im.convert("RGB")

    resize_target = int(image_size * 256 / 224)
    width, height = image.size
    scale = resize_target / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)

    width, height = image.size
    left = (width - image_size) // 2
    top = (height - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return array.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with optional temperature. Max-subtracted for overflow safety.

    `temperature` applies the calibration fitted by cropguard.evaluation.calibrate. T < 1
    sharpens, T > 1 softens; T == 1 is a no-op. Because dividing by a positive scalar is
    monotonic it cannot reorder logits, so the predicted class is identical either way.
    """
    scaled = np.asarray(logits, dtype=np.float32) / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class CropGuardModel:
    """ONNX Runtime inference with optional temperature calibration.

    `temperature` comes from cropguard.evaluation.calibrate, fitted on the validation split.
    Without it the API reports raw softmax, which for this model is systematically
    *under*-confident: label smoothing bounds the achievable output at ~0.90 while the model
    is 99.11% accurate. Serving that number unmodified means telling a user "90% sure" about a
    prediction that is right 99.5% of the time.
    """

    def __init__(
        self,
        model_path: Path,
        classes_path: Path,
        model_version: str,
        temperature: float = 1.0,
    ) -> None:
        import onnxruntime as ort

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        with open(classes_path, encoding="utf-8") as f:
            self.class_names: list[str] = json.load(f)
        self.model_version = model_version
        self.temperature = float(temperature)

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        inputs = preprocess(image_bytes)
        logits = self.session.run(["logits"], {"input": inputs})[0][0]
        probs = softmax(logits, self.temperature)

        top_indices = np.argsort(probs)[::-1][:top_k]
        # Normalized predictive entropy in [0, 1]. Measures spread in this one softmax, so it
        # cannot flag an input unlike anything in training - MC-dropout or an ensemble would.
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs)))

        return {
            "predicted_class": self.class_names[int(top_indices[0])],
            "confidence": float(probs[top_indices[0]]),
            "top_k": [
                {"class_name": self.class_names[int(i)], "probability": float(probs[i])}
                for i in top_indices
            ],
            "uncertainty": entropy,
            "calibrated": self.temperature != 1.0,
            "model_version": self.model_version,
        }
