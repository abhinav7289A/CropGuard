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

    model_path: Path = Path("models/cropguard.int8.onnx")
    classes_path: Path = Path("configs/classes.json")
    model_version: str = "v0.1.0-baseline"
    cors_origins: str = "http://localhost:5173"  # comma-separated
    max_upload_mb: int = 10
    min_resolution: int = 128


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
                settings.model_path, settings.classes_path, settings.model_version
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
        "model_error": _model_error,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "api_version": __version__,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
