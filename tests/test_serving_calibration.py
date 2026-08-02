"""The serving path must apply the fitted temperature — and must not change predictions.

Without it the API reports raw softmax, which for this model is systematically
*under*-confident: label smoothing caps output at ~0.90 while the model is 99.11% accurate.
Serving that unmodified tells a user "90% sure" about a prediction that is right 99.5% of the
time.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from cropguard.serving.model_loader import CropGuardModel, softmax  # noqa: E402

CLASSES = ["Potato___Late_blight", "Tomato___Early_blight", "Tomato___healthy"]


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def model_files(tmp_path_factory) -> tuple[Path, Path]:
    net = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, len(CLASSES))
    ).eval()
    directory = tmp_path_factory.mktemp("serve")
    model_path = directory / "m.onnx"
    torch.onnx.export(
        net,
        torch.randn(1, 3, 224, 224),
        str(model_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
        verbose=False,
    )
    classes_path = directory / "classes.json"
    classes_path.write_text(json.dumps(CLASSES), encoding="utf-8")
    return model_path, classes_path


# --- the softmax primitive --------------------------------------------------------------


def test_temperature_below_one_sharpens():
    logits = np.array([3.0, 1.0, 0.0], dtype=np.float32)
    assert softmax(logits, 0.5).max() > softmax(logits, 1.0).max()


def test_temperature_above_one_softens():
    logits = np.array([3.0, 1.0, 0.0], dtype=np.float32)
    assert softmax(logits, 2.0).max() < softmax(logits, 1.0).max()


def test_temperature_one_is_a_no_op():
    logits = np.array([3.0, 1.0, 0.0], dtype=np.float32)
    assert np.allclose(softmax(logits, 1.0), softmax(logits))


def test_softmax_stays_normalized_at_any_temperature():
    logits = np.array([1000.0, 999.0, 998.0], dtype=np.float32)
    for temperature in (0.1, 0.591, 1.0, 5.0):
        probabilities = softmax(logits, temperature)
        assert np.isfinite(probabilities).all()
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


# --- the loader -------------------------------------------------------------------------


def test_temperature_never_changes_the_prediction(model_files):
    """The safety property. Dividing by a positive scalar is monotonic, so the argmax is
    fixed — which is what makes it safe to apply to a model already serving traffic."""
    model_path, classes_path = model_files
    image = _image_bytes()

    raw = CropGuardModel(model_path, classes_path, "v", temperature=1.0).predict(image)
    calibrated = CropGuardModel(model_path, classes_path, "v", temperature=0.591).predict(image)

    assert raw["predicted_class"] == calibrated["predicted_class"]
    assert [e["class_name"] for e in raw["top_k"]] == [e["class_name"] for e in calibrated["top_k"]]


def test_calibration_sharpens_reported_confidence(model_files):
    model_path, classes_path = model_files
    image = _image_bytes()

    raw = CropGuardModel(model_path, classes_path, "v", temperature=1.0).predict(image)
    calibrated = CropGuardModel(model_path, classes_path, "v", temperature=0.591).predict(image)

    assert calibrated["confidence"] > raw["confidence"]
    assert calibrated["uncertainty"] < raw["uncertainty"]


def test_response_declares_whether_it_is_calibrated(model_files):
    """A consumer must be able to tell whether `confidence` has been calibrated."""
    model_path, classes_path = model_files
    image = _image_bytes()

    assert CropGuardModel(model_path, classes_path, "v").predict(image)["calibrated"] is False
    assert (
        CropGuardModel(model_path, classes_path, "v", temperature=0.591).predict(image)[
            "calibrated"
        ]
        is True
    )


def test_probabilities_still_sum_to_one_after_calibration(model_files):
    model_path, classes_path = model_files
    result = CropGuardModel(model_path, classes_path, "v", temperature=0.591).predict(
        _image_bytes(), top_k=len(CLASSES)
    )
    assert sum(e["probability"] for e in result["top_k"]) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_temperature_is_rejected(model_files, bad):
    """T <= 0 would flip or destroy the ordering — fail loudly rather than serve nonsense."""
    model_path, classes_path = model_files
    with pytest.raises(ValueError, match="must be positive"):
        CropGuardModel(model_path, classes_path, "v", temperature=bad)
