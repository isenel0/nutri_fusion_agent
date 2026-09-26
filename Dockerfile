# Nutri Fusion Agent - FastAPI backend (CPU image)
#
# Build:  docker build -t nutri-fusion-agent .
# Run:    docker run --rm -p 8000:8000 \
#           -v "$(pwd)/../evaluation_dataset/barcode_nutrition_cache.json:/app/data/barcode_nutrition_cache.json:ro" \
#           nutri-fusion-agent
#
# Ollama is expected on the host (ollama serve). If it is unreachable the
# system falls back to deterministic fusion rules.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libzbar0: barcode decoding (pyzbar)
# libgl1, libglib2.0-0: OpenCV runtime libs pulled in by ultralytics
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzbar0 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first, so requirements.txt doesn't pull the multi-GB CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown app:app /app/data

# Source code + bundled model weights (YOLO, Swin, BERT) under src/agents/
COPY --chown=app:app src/ src/
COPY --chown=app:app scripts/ scripts/
# Nutrition prior database used by NutritionLookup
COPY --chown=app:app nutrition5k/metadata/nutrition_db.json nutrition5k/metadata/nutrition_db.json

USER app

ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_FUSION_MODEL=qwen3.5:9b \
    BARCODE_NUTRITION_CACHE_PATH=/app/data/barcode_nutrition_cache.json \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    HF_HUB_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
