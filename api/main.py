"""
FastAPI Backend — Medical Imaging RAG Clinical Decision Support

Endpoints:
    GET  /              — Landing page with interactive demo
    GET  /health        — Service health check
    GET  /stats         — Corpus and index statistics
    GET  /conditions    — List of supported conditions
    POST /retrieve      — Retrieve relevant chunks (condition-focused)
    POST /query         — Full RAG pipeline: retrieve + LLM summary
"""

import os
import json
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk,
    HealthResponse,
    StatsResponse,
    CONDITIONS,
    CONDITION_CONTEXT,
)
from api.llm_provider import generate_clinical_summary
from rag.retrieval_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

pipeline: RAGPipeline = None
corpus_size: int = 0
chunk_count: int = 0


def load_pipeline():
    global pipeline, corpus_size, chunk_count
    logger.info("Initializing RAG pipeline...")
    start = time.perf_counter()

    pipeline = RAGPipeline(
        model_key="minilm",
        strategy_key="recursive_paragraph",
        k=7,
        use_postgres=False,
    )

    docs_path = Path("pipelines/pubmed_cache/documents.json")
    if not docs_path.exists():
        raise FileNotFoundError(f"Corpus not found: {docs_path}")

    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    corpus_size = len(documents)
    logger.info(f"Loaded {corpus_size} documents from cache")
    pipeline.build_index(documents)
    chunk_count = len(pipeline._chunk_texts)

    elapsed = time.perf_counter() - start
    logger.info(f"Pipeline ready: {corpus_size} docs, {chunk_count} chunks, {elapsed:.1f}s")


def build_focused_query(query: str, condition: str) -> str:
    """Augment the user query with condition-specific terms for better retrieval."""
    if condition and condition in CONDITION_CONTEXT:
        context = CONDITION_CONTEXT[condition]
        return f"{query} {context}"
    return query


_landing_html = ""
_html_path = Path(__file__).parent / "index.html"
if _html_path.exists():
    _landing_html = _html_path.read_text(encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pipeline()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Medical Imaging RAG Clinical Decision Support",
    description="RAG-based clinical decision support with condition-focused hybrid retrieval and LLM-generated summaries.",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return _landing_html


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        corpus_loaded=corpus_size > 0,
        corpus_size=corpus_size,
        chunk_count=chunk_count,
    )


@app.get("/conditions")
async def list_conditions():
    """List all supported conditions with retrieval context."""
    return {"conditions": CONDITIONS}


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start = time.perf_counter()

    focused_query = build_focused_query(request.query, request.condition)
    logger.info(f"Retrieval: condition={request.condition or 'none'}, query={request.query[:60]}")

    try:
        docs, retrieval_ms = pipeline.retrieve(focused_query)
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
        condition=request.condition or "none",
        chunks=chunks,
        retrieval_latency_ms=round(retrieval_ms, 2),
        total_latency_ms=round(total_ms, 2),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    total_start = time.perf_counter()

    condition = request.condition or request.cnn_prediction
    focused_query = build_focused_query(request.query, condition)
    logger.info(f"Full query: condition={condition}, query={request.query[:60]}")

    try:
        docs, retrieval_ms = pipeline.retrieve(focused_query)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    chunks = [
        RetrievedChunk(
            chunk_text=doc.chunk_text,
            similarity=round(doc.similarity, 4),
            doc_id=doc.doc_id,
            chunk_index=doc.chunk_index,
        )
        for doc in docs
    ]

    gen_start = time.perf_counter()
    try:
        chunk_dicts = [
            {"doc_id": c.doc_id, "chunk_text": c.chunk_text, "similarity": c.similarity}
            for c in chunks
        ]
        summary = generate_clinical_summary(
            query=request.query,
            cnn_prediction=condition if condition != "unknown" else "not specified",
            confidence=request.confidence,
            chunks=chunk_dicts,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        summary = f"*Summary generation failed: {str(e)}*"

    gen_ms = (time.perf_counter() - gen_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    return QueryResponse(
        query=request.query,
        cnn_prediction=request.cnn_prediction,
        confidence=request.confidence,
        condition=condition or "none",
        chunks=chunks,
        generated_response=summary,
        retrieval_latency_ms=round(retrieval_ms, 2),
        generation_latency_ms=round(gen_ms, 2),
        total_latency_ms=round(total_ms, 2),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return StatsResponse(
        corpus_size=corpus_size,
        chunk_count=chunk_count,
        embedding_model=pipeline.model_key,
        chunking_strategy=pipeline.strategy_key,
        retrieval_k=pipeline.k,
        conditions=CONDITIONS,
    )
