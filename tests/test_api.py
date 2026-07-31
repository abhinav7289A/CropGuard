from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("pydantic_settings")
from fastapi.testclient import TestClient  # noqa: E402

from cropguard.serving import app as app_module  # noqa: E402


@pytest.fixture(autouse=True)
def unloaded_model(monkeypatch, tmp_path):
    """Reset the module-level model cache between tests and point at a missing model."""
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "_model_error", None)
    monkeypatch.setattr(app_module.settings, "model_path", tmp_path / "absent.onnx")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


def _encode(size=(256, 256), fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "green").save(buffer, format=fmt)
    return buffer.getvalue()


def test_health_is_degraded_without_a_model(client):
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["uptime_seconds"] >= 0


def test_predict_returns_503_when_model_is_missing(client):
    response = client.post("/predict", files={"file": ("leaf.jpg", _encode(), "image/jpeg")})
    assert response.status_code == 503
    assert "Model unavailable" in response.json()["detail"]


def test_predict_rejects_undersized_image(client):
    response = client.post(
        "/predict", files={"file": ("tiny.jpg", _encode((64, 64)), "image/jpeg")}
    )
    assert response.status_code == 422


def test_predict_rejects_unsupported_format(client):
    response = client.post(
        "/predict", files={"file": ("leaf.bmp", _encode(fmt="BMP"), "image/bmp")}
    )
    assert response.status_code == 415


def test_predict_rejects_undecodable_bytes(client):
    response = client.post(
        "/predict", files={"file": ("leaf.jpg", b"definitely not jpeg", "image/jpeg")}
    )
    assert response.status_code == 422


def test_predict_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(app_module.settings, "max_upload_mb", 0)
    response = client.post("/predict", files={"file": ("leaf.jpg", _encode(), "image/jpeg")})
    assert response.status_code == 413


def test_metrics_endpoint_exposes_prometheus_counters(client):
    client.post("/predict", files={"file": ("tiny.jpg", _encode((64, 64)), "image/jpeg")})
    body = client.get("/metrics").text
    assert "cropguard_requests_total" in body
    assert 'endpoint="/predict"' in body


# --- End-to-end serving path against a real (tiny) ONNX graph -------------------------


@pytest.fixture
def onnx_model(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")

    classes = ["Potato___Late_blight", "Tomato___Early_blight", "Tomato___healthy"]
    net = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, len(classes))
    ).eval()

    model_path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        net,
        torch.randn(1, 3, 224, 224),
        str(model_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(classes), encoding="utf-8")
    return model_path, classes_path, classes


def test_predict_end_to_end(client, monkeypatch, onnx_model):
    model_path, classes_path, classes = onnx_model
    monkeypatch.setattr(app_module.settings, "model_path", model_path)
    monkeypatch.setattr(app_module.settings, "classes_path", classes_path)

    response = client.post("/predict", files={"file": ("leaf.jpg", _encode(), "image/jpeg")})
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_class"] in classes
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 3
    # top_k must be sorted by probability, descending, and consistent with the winner.
    probabilities = [entry["probability"] for entry in body["top_k"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert body["top_k"][0]["class_name"] == body["predicted_class"]
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= body["uncertainty"] <= 1.0
    assert body["latency_ms"] > 0


def test_health_is_healthy_once_the_model_loads(client, monkeypatch, onnx_model):
    model_path, classes_path, _ = onnx_model
    monkeypatch.setattr(app_module.settings, "model_path", model_path)
    monkeypatch.setattr(app_module.settings, "classes_path", classes_path)

    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_error"] is None
