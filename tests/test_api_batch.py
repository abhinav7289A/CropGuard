"""Batch endpoint tests.

The design decision under test: a single bad image must not discard the rest of the batch.
Failing the whole request on one malformed upload makes the endpoint unusable for its actual
purpose, which is processing a folder of field photos where a few are always broken.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("pydantic_settings")
from fastapi.testclient import TestClient  # noqa: E402

from cropguard.serving import app as app_module  # noqa: E402

CLASSES = ["Potato___Late_blight", "Tomato___Early_blight", "Tomato___healthy"]


def _encode(size=(256, 256), fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "green").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path) -> TestClient:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")

    net = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, len(CLASSES))
    ).eval()
    model_path = tmp_path / "m.onnx"
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
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(CLASSES), encoding="utf-8")

    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "_model_error", None)
    monkeypatch.setattr(app_module.settings, "model_path", Path(model_path))
    monkeypatch.setattr(app_module.settings, "classes_path", Path(classes_path))
    return TestClient(app_module.app)


def _files(count: int):
    return [("files", (f"leaf{i}.jpg", _encode(), "image/jpeg")) for i in range(count)]


def test_batch_returns_one_result_per_image(client):
    response = client.post("/predict/batch", files=_files(3))
    assert response.status_code == 200

    body = response.json()
    assert body["count"] == 3
    assert body["succeeded"] == 3
    assert body["failed"] == 0
    assert len(body["results"]) == 3
    assert all("predicted_class" in r for r in body["results"])
    assert body["latency_ms"] > 0


def test_results_carry_filenames_so_they_can_be_matched_up(client):
    response = client.post("/predict/batch", files=_files(2))
    assert [r["filename"] for r in response.json()["results"]] == ["leaf0.jpg", "leaf1.jpg"]


def test_one_bad_image_does_not_sink_the_batch(client):
    files = [
        ("files", ("good.jpg", _encode(), "image/jpeg")),
        ("files", ("tiny.jpg", _encode((64, 64)), "image/jpeg")),  # below min resolution
        ("files", ("also_good.jpg", _encode(), "image/jpeg")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200

    body = response.json()
    assert body["succeeded"] == 2
    assert body["failed"] == 1

    failed = [r for r in body["results"] if "error" in r][0]
    assert failed["filename"] == "tiny.jpg"
    assert failed["status"] == 422
    # Order is preserved, so a caller can zip results back to inputs.
    assert [r["filename"] for r in body["results"]] == ["good.jpg", "tiny.jpg", "also_good.jpg"]


def test_oversized_batch_is_rejected(client, monkeypatch):
    monkeypatch.setattr(app_module.settings, "max_batch_size", 2)
    response = client.post("/predict/batch", files=_files(3))
    assert response.status_code == 413
    assert "exceeds limit" in response.json()["detail"]


def test_batch_is_recorded_separately_in_metrics(client):
    client.post("/predict/batch", files=_files(2))
    body = client.get("/metrics").text
    assert 'endpoint="/predict/batch"' in body


def test_batch_returns_503_when_no_model_is_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module.settings, "model_path", tmp_path / "absent.onnx")
    response = TestClient(app_module.app).post("/predict/batch", files=_files(2))
    assert response.status_code == 503
