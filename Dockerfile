# Serving image - CPU-only ONNX Runtime, no torch. Target: well under 500MB.
#
# Build with the model baked in (recommended for Render's free tier, where the instance
# spins down after 15 minutes idle and would otherwise re-download on every cold start):
#     docker build --build-arg CROPGUARD_HF_REPO=<user>/cropguard-models -t cropguard .
#
# Build without it - the API starts degraded and fetch_model.py pulls the weights at boot:
#     docker build -t cropguard .
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dependencies first: this layer is cached unless pyproject.toml changes, so application
# edits do not trigger a full reinstall on every build.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[serve]" huggingface_hub

COPY configs/classes.json configs/classes.json
COPY scripts/fetch_model.py scripts/fetch_model.py

# Bake the weights in at build time when a repo is supplied. `|| true` keeps the build
# working without one - the model is then fetched at startup instead.
#
# NOTE: do not reintroduce `COPY models*/ models/` here. models/ is gitignored, so it does
# not exist in a clean checkout, and a COPY whose glob matches nothing fails the build
# outright rather than being skipped.
# Defaults rather than required build args: Render never forwards Blueprint env vars into the
# build, so a default is the only way to bake the weights in there. Override locally with
# --build-arg when testing a different model.
#
# fp32, not INT8: dynamic quantization rewrites every Conv into ConvInteger, which ONNX
# Runtime's CPU backend has no optimized kernel for. Measured 1567 ms/img against 19 ms/img
# for fp32 - a 75x regression in exchange for 4x less disk. Static quantization (QLinearConv)
# is the correct fix and is not built yet.
ARG CROPGUARD_HF_REPO="XiElonMAsk/cropguard-models"
ARG CROPGUARD_MODEL_FILE="cropguard.onnx"
ENV CROPGUARD_MODEL_PATH=/app/models/cropguard.onnx
RUN mkdir -p models && \
    if [ -n "$CROPGUARD_HF_REPO" ]; then \
        CROPGUARD_HF_REPO="$CROPGUARD_HF_REPO" \
        CROPGUARD_MODEL_FILE="$CROPGUARD_MODEL_FILE" \
        python scripts/fetch_model.py || echo "build-time model fetch failed; will retry at startup"; \
    else \
        echo "no CROPGUARD_HF_REPO supplied - model will be fetched at startup"; \
    fi

# Run as a non-root user; nothing here needs write access outside /app/models.
RUN useradd --create-home --uid 10001 cropguard && chown -R cropguard:cropguard /app
USER cropguard

EXPOSE 8000

# /health reports "degraded" (HTTP 200) when no model is loaded, so this checks the process
# is serving, not that a model exists - which is what lets the container deploy before the
# first model is trained.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "python scripts/fetch_model.py && uvicorn cropguard.serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
