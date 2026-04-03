# ============================================================
# Medical Imaging RAG Clinical Decision Support — Dockerfile
# ============================================================
# Multi-stage build to keep final image smaller.
# Stage 1: Install dependencies
# Stage 2: Copy app code and run
# ============================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ─────────────────────────────
# Copy requirements first (Docker caches this layer — if requirements
# don't change, it skips reinstalling everything on rebuild)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────
# Copy only what the API needs to run (not data/, models/, etc.)
COPY api/ api/
COPY rag/ rag/
COPY evaluation/ evaluation/
COPY pipelines/pubmed_cache/documents.json pipelines/pubmed_cache/documents.json
COPY models/checkpoints/efficientnet_b3_multilabel_best.pth models/checkpoints/
COPY models/checkpoints/optimal_thresholds.json models/checkpoints/

# ── Expose port ─────────────────────────────────────────────
# Cloud Run uses PORT env variable, default 8080
ENV PORT=8080
EXPOSE ${PORT}

# ── Run the API ─────────────────────────────────────────────
# uvicorn serves the FastAPI app
# --host 0.0.0.0 makes it accessible outside the container
# Cloud Run sets PORT env variable automatically
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
