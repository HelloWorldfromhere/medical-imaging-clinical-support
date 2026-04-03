"""
FastAPI Backend — Medical Imaging RAG Clinical Decision Support
Full pipeline: Image Upload → CNN Prediction → RAG Retrieval → LLM Summary
"""

import os
import json
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.schemas import (
    QueryRequest, QueryResponse, RetrieveRequest, RetrieveResponse,
    RetrievedChunk, HealthResponse, StatsResponse, PredictResponse,
    AnalyzeResponse, ConditionPrediction, CONDITIONS, CONDITION_CONTEXT,
)
from api.llm_provider import generate_clinical_summary
from api.cnn_inference import load_model as load_cnn, predict_conditions, is_model_loaded
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

    pipeline = RAGPipeline(model_key="minilm", strategy_key="recursive_paragraph", k=7, use_postgres=False)

    docs_path = Path("pipelines/pubmed_cache/documents.json")
    if not docs_path.exists():
        raise FileNotFoundError(f"Corpus not found: {docs_path}")

    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    corpus_size = len(documents)
    pipeline.build_index(documents)
    chunk_count = len(pipeline._chunk_texts)

    elapsed = time.perf_counter() - start
    logger.info(f"Pipeline ready: {corpus_size} docs, {chunk_count} chunks, {elapsed:.1f}s")

    logger.info("Loading CNN model...")
    load_cnn()


def build_focused_query(query: str, condition: str) -> str:
    if condition and condition in CONDITION_CONTEXT:
        return f"{query} {CONDITION_CONTEXT[condition]}"
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
    description="Full pipeline: X-ray upload → CNN classification → RAG retrieval → LLM clinical summary",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return _landing_html


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", corpus_loaded=corpus_size > 0, corpus_size=corpus_size, chunk_count=chunk_count, cnn_model_loaded=is_model_loaded())


@app.get("/conditions")
async def list_conditions():
    return {"conditions": CONDITIONS, "model_loaded": is_model_loaded()}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Upload a chest X-ray image and get condition predictions."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (PNG, JPEG)")

    start = time.perf_counter()
    image_bytes = await file.read()

    result = predict_conditions(image_bytes)
    latency = (time.perf_counter() - start) * 1000

    return PredictResponse(
        predictions=[ConditionPrediction(**p) for p in result["predictions"]],
        model_loaded=result["model_loaded"],
        needs_manual_selection=result.get("needs_manual_selection", False),
        inference_latency_ms=round(latency, 2),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    query: str = Form(default=""),
):
    """
    Full pipeline: upload X-ray → CNN prediction → RAG retrieval → LLM summary.
    Optionally provide additional clinical context in the query field.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    total_start = time.perf_counter()

    # Step 1: CNN prediction
    image_bytes = await file.read()
    cnn_start = time.perf_counter()
    cnn_result = predict_conditions(image_bytes)
    predictions = cnn_result["predictions"]
    cnn_ms = (time.perf_counter() - cnn_start) * 1000

    detected = [p for p in predictions if p["detected"]]
    detected_names = [p["condition"] for p in detected]
    top_condition = predictions[0]["condition"] if predictions else "unknown"
    top_confidence = predictions[0]["probability"] if predictions else 0.0

    # Step 2: Build query from CNN results + user context
    if not query:
        if detected_names:
            query = f"Chest X-ray findings suggest: {', '.join(detected_names)}. Provide clinical guidance."
        else:
            query = f"Chest X-ray analysis. Top finding: {top_condition} ({top_confidence:.0%} confidence)."

    condition_terms = " ".join(CONDITION_CONTEXT.get(c, "") for c in detected_names[:3]) if detected_names else CONDITION_CONTEXT.get(top_condition, "")
    focused_query = f"{query} {condition_terms}"

    # Step 3: RAG retrieval
    ret_start = time.perf_counter()
    try:
        docs, retrieval_ms = pipeline.retrieve(focused_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    chunks = [RetrievedChunk(chunk_text=d.chunk_text, similarity=round(d.similarity, 4), doc_id=d.doc_id, chunk_index=d.chunk_index) for d in docs]

    # Step 4: LLM summary
    gen_start = time.perf_counter()
    try:
        chunk_dicts = [{"doc_id": c.doc_id, "chunk_text": c.chunk_text, "similarity": c.similarity} for c in chunks]
        cnn_desc = ", ".join([f"{p['condition']} ({p['probability']:.0%})" for p in detected[:5]]) if detected else f"{top_condition} ({top_confidence:.0%})"
        summary = generate_clinical_summary(query=query, cnn_prediction=cnn_desc, confidence=top_confidence, chunks=chunk_dicts)
    except Exception as e:
        summary = f"*Summary generation failed: {str(e)}*"

    gen_ms = (time.perf_counter() - gen_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    return AnalyzeResponse(
        predictions=[ConditionPrediction(**p) for p in predictions],
        detected_conditions=detected_names,
        query=query,
        chunks=chunks,
        generated_response=summary,
        model_loaded=is_model_loaded(),
        inference_latency_ms=round(cnn_ms, 2),
        retrieval_latency_ms=round(retrieval_ms, 2),
        generation_latency_ms=round(gen_ms, 2),
        total_latency_ms=round(total_ms, 2),
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    start = time.perf_counter()
    focused_query = build_focused_query(request.query, request.condition)
    try:
        docs, retrieval_ms = pipeline.retrieve(focused_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    total_ms = (time.perf_counter() - start) * 1000
    chunks = [RetrievedChunk(chunk_text=d.chunk_text, similarity=round(d.similarity, 4), doc_id=d.doc_id, chunk_index=d.chunk_index) for d in docs]
    return RetrieveResponse(query=request.query, condition=request.condition or "none", chunks=chunks, retrieval_latency_ms=round(retrieval_ms, 2), total_latency_ms=round(total_ms, 2))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    total_start = time.perf_counter()
    condition = request.condition or request.cnn_prediction
    focused_query = build_focused_query(request.query, condition)
    try:
        docs, retrieval_ms = pipeline.retrieve(focused_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    chunks = [RetrievedChunk(chunk_text=d.chunk_text, similarity=round(d.similarity, 4), doc_id=d.doc_id, chunk_index=d.chunk_index) for d in docs]
    gen_start = time.perf_counter()
    try:
        chunk_dicts = [{"doc_id": c.doc_id, "chunk_text": c.chunk_text, "similarity": c.similarity} for c in chunks]
        summary = generate_clinical_summary(query=request.query, cnn_prediction=condition if condition != "unknown" else "not specified", confidence=request.confidence, chunks=chunk_dicts)
    except Exception as e:
        summary = f"*Summary generation failed: {str(e)}*"
    gen_ms = (time.perf_counter() - gen_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    return QueryResponse(query=request.query, cnn_prediction=request.cnn_prediction, confidence=request.confidence, condition=condition or "none", chunks=chunks, generated_response=summary, retrieval_latency_ms=round(retrieval_ms, 2), generation_latency_ms=round(gen_ms, 2), total_latency_ms=round(total_ms, 2))


@app.get("/stats", response_model=StatsResponse)
async def stats():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return StatsResponse(corpus_size=corpus_size, chunk_count=chunk_count, embedding_model=pipeline.model_key, chunking_strategy=pipeline.strategy_key, retrieval_k=pipeline.k, conditions=CONDITIONS, cnn_model_loaded=is_model_loaded())
