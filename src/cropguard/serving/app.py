"""CropGuard prediction API.

Run locally:
    uvicorn cropguard.serving.app:app --reload --port 8000

Env vars (see Settings): CROPGUARD_MODEL_PATH, CROPGUARD_CLASSES_PATH,
CROPGUARD_MODEL_VERSION, CROPGUARD_CORS_ORIGINS.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic_settings import BaseSettings, SettingsConfigDict

from cropguard import __version__
from cropguard.serving.model_loader import CropGuardModel


class Settings(BaseSettings):
    # protected_namespaces=() so pydantic permits the model_* field names used by the env vars.
    model_config = SettingsConfigDict(env_prefix="CROPGUARD_", protected_namespaces=())

    # fp32 by default: dynamically-quantized INT8 rewrites Conv into ConvInteger, which ONNX
    # Runtime's CPU backend has no optimized kernel for (measured 1567 ms/img vs 19 ms/img).
    model_path: Path = Path("models/cropguard.onnx")
    classes_path: Path = Path("configs/classes.json")
    model_version: str = "v0.1.0-baseline"
    # Comma-separated allowlist. "*" is deliberately not the default: CORS is the only thing
    # stopping an arbitrary page from calling this API with a visitor's browser.
    cors_origins: str = "http://localhost:5173,http://localhost:8501"
    max_upload_mb: int = 10
    min_resolution: int = 128
    max_batch_size: int = 8

    # Temperature from cropguard.evaluation.calibrate, fitted on validation. 1.0 = raw softmax.
    # This model is under-confident without it: label smoothing caps output at ~0.90 while the
    # model is 99.11% accurate, so the honest fitted value is 0.591.
    temperature: float = 1.0


settings = Settings()
app = FastAPI(title="CropGuard API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUESTS = Counter(
    "cropguard_requests_total", "Total requests", ["endpoint", "status", "model_version"]
)
LATENCY = Histogram("cropguard_request_duration_seconds", "Request duration", ["endpoint"])
CONFIDENCE = Histogram(
    "cropguard_prediction_confidence",
    "Prediction confidence",
    ["predicted_class"],
    buckets=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
)

START_TIME = time.time()
_model: CropGuardModel | None = None
_model_error: str | None = None


def get_model() -> CropGuardModel:
    global _model, _model_error
    if _model is None:
        try:
            _model = CropGuardModel(
                settings.model_path,
                settings.classes_path,
                settings.model_version,
                temperature=settings.temperature,
            )
        except Exception as exc:  # model not trained/downloaded yet
            _model_error = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Model unavailable ({settings.model_path}): {exc}",
            ) from exc
    return _model


def _validate_image(data: bytes) -> None:
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {settings.max_upload_mb}MB limit")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in {"JPEG", "PNG"}:
                raise HTTPException(415, f"Unsupported format {im.format}; use JPEG or PNG")
            if min(im.size) < settings.min_resolution:
                raise HTTPException(
                    422, f"Image too small ({im.size}); min {settings.min_resolution}px"
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, "Could not decode image") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    start = time.time()
    data = await file.read()
    try:
        _validate_image(data)
        result = get_model().predict(data)
    except HTTPException as exc:
        REQUESTS.labels("/predict", str(exc.status_code), settings.model_version).inc()
        raise
    duration = time.time() - start
    LATENCY.labels("/predict").observe(duration)
    REQUESTS.labels("/predict", "200", settings.model_version).inc()
    CONFIDENCE.labels(result["predicted_class"]).observe(result["confidence"])
    result["latency_ms"] = round(duration * 1000, 1)
    return result


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict:
    """Several images in one request.

    Capped at `max_batch_size`: each image is a full forward pass, and on the free tier's
    0.1 vCPU a large batch would hold the single worker long enough to look like an outage.
    Validation failures are reported per image rather than failing the whole request, so one
    bad upload does not discard the rest.
    """
    start = time.time()
    if len(files) > settings.max_batch_size:
        REQUESTS.labels("/predict/batch", "413", settings.model_version).inc()
        raise HTTPException(
            413, f"Batch of {len(files)} exceeds limit of {settings.max_batch_size}"
        )

    model = get_model()  # 503s the whole request if no model - nothing to partially succeed at
    results: list[dict] = []
    for upload in files:
        data = await upload.read()
        try:
            _validate_image(data)
            result = model.predict(data)
        except HTTPException as exc:
            results.append(
                {"filename": upload.filename, "error": exc.detail, "status": exc.status_code}
            )
            continue
        result["filename"] = upload.filename
        results.append(result)
        CONFIDENCE.labels(result["predicted_class"]).observe(result["confidence"])

    duration = time.time() - start
    LATENCY.labels("/predict/batch").observe(duration)
    REQUESTS.labels("/predict/batch", "200", settings.model_version).inc()
    succeeded = sum(1 for r in results if "error" not in r)
    return {
        "results": results,
        "count": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "latency_ms": round(duration * 1000, 1),
    }


@app.get("/health")
def health() -> dict:
    model_loaded = _model is not None
    if not model_loaded and settings.model_path.exists():
        try:
            get_model()
            model_loaded = True
        except HTTPException:
            pass
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": settings.model_version,
        "calibrated": settings.temperature != 1.0,
        "temperature": settings.temperature,
        "model_error": _model_error,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "api_version": __version__,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
