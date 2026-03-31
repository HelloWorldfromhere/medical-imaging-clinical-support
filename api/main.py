"""
FastAPI Backend — Medical Imaging RAG Clinical Decision Support
Wraps the existing RAGPipeline with REST API endpoints.

Endpoints:
    GET  /health          — Service health check
    POST /retrieve         — Retrieve relevant chunks for a query
    POST /query            — Full RAG pipeline: retrieve + generate
    GET  /stats            — Corpus and index statistics

Run locally:
    uvicorn api.main:app --reload --port 8000

Then visit: http://localhost:8000/docs for interactive API docs
"""

import os
import json
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk,
    HealthResponse,
    StatsResponse,
)
from rag.retrieval_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Global pipeline instance ────────────────────────────────────────────────

pipeline: RAGPipeline = None
corpus_size: int = 0
chunk_count: int = 0


def load_pipeline():
    """Load documents and build in-memory index."""
    global pipeline, corpus_size, chunk_count

    logger.info("Initializing RAG pipeline...")
    start = time.perf_counter()

    # Use best config from evaluation (MPNet, recursive 800/80)
    pipeline = RAGPipeline(
        model_key="minilm",  # Change to your best model key
        strategy_key="recursive_paragraph",  # Change to your best strategy key
        k=7,
        use_postgres=False,
    )

    # Load documents from JSON cache
    docs_path = Path("pipelines/pubmed_cache/documents.json")
    if not docs_path.exists():
        raise FileNotFoundError(f"Corpus not found: {docs_path}")

    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    corpus_size = len(documents)
    logger.info(f"Loaded {corpus_size} documents from cache")

    # Build in-memory index
    pipeline.build_index(documents)
    chunk_count = len(pipeline._chunk_texts)

    elapsed = time.perf_counter() - start
    logger.info(f"Pipeline ready: {corpus_size} docs, {chunk_count} chunks, {elapsed:.1f}s")


# ── App lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, cleanup on shutdown."""
    load_pipeline()
    yield
    logger.info("Shutting down...")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Imaging RAG Clinical Decision Support",
    description=(
        "A RAG-based clinical decision support system that retrieves relevant "
        "medical literature based on chest X-ray findings and clinical queries. "
        "Designed as a portfolio project demonstrating hybrid BM25 + vector "
        "retrieval with systematic evaluation across embedding models and "
        "chunking strategies."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend or demo tools to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check."""
    return HealthResponse(
        status="healthy",
        corpus_loaded=corpus_size > 0,
        corpus_size=corpus_size,
        chunk_count=chunk_count,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """
    Retrieve relevant medical literature chunks for a clinical query.

    Returns the top-k most relevant chunks with similarity scores.
    Uses hybrid BM25 + vector retrieval with RRF fusion.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start = time.perf_counter()

    try:
        docs, retrieval_ms = pipeline.retrieve(request.query)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    total_ms = (time.perf_counter() - start) * 1000

    chunks = [
        RetrievedChunk(
            chunk_text=doc.chunk_text,
            similarity=round(doc.similarity, 4),
            doc_id=doc.doc_id,
            chunk_index=doc.chunk_index,
        )
        for doc in docs
    ]

    return RetrieveResponse(
        query=request.query,
        chunks=chunks,
        retrieval_latency_ms=round(retrieval_ms, 2),
        total_latency_ms=round(total_ms, 2),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Full RAG pipeline: retrieve relevant literature + generate clinical summary.

    Accepts a clinical query and optional CNN prediction results.
    Returns retrieved chunks and a generated clinical summary.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        response = pipeline.query(
            query=request.query,
            cnn_prediction=request.cnn_prediction,
            confidence=request.confidence,
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    chunks = [
        RetrievedChunk(
            chunk_text=doc.chunk_text,
            similarity=round(doc.similarity, 4),
            doc_id=doc.doc_id,
            chunk_index=doc.chunk_index,
        )
        for doc in response.retrieved_docs
    ]

    return QueryResponse(
        query=response.query,
        cnn_prediction=response.cnn_prediction,
        confidence=response.confidence,
        chunks=chunks,
        generated_response=response.generated_response,
        retrieval_latency_ms=response.retrieval_latency_ms,
        generation_latency_ms=response.generation_latency_ms,
        total_latency_ms=response.total_latency_ms,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Corpus and index statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return StatsResponse(
        corpus_size=corpus_size,
        chunk_count=chunk_count,
        embedding_model=pipeline.model_key,
        chunking_strategy=pipeline.strategy_key,
        retrieval_k=pipeline.k,
    )
