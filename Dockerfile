# Serving image — CPU-only ONNX Runtime, no torch. Target: well under 500MB.
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[serve]" huggingface_hub

COPY configs/classes.json configs/classes.json
COPY scripts/fetch_model.py scripts/fetch_model.py
# Bake the model in if present at build time (models/ may be empty pre-training;
# then fetch_model.py pulls it from HF Hub at startup instead).
COPY models*/ models/

EXPOSE 8000
CMD ["sh", "-c", "python scripts/fetch_model.py && uvicorn cropguard.serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
