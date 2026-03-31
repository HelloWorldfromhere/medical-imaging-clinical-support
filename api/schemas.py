"""
Pydantic models for API request and response schemas.
"""

from pydantic import BaseModel, Field


# ── Request models ──────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    """Request to retrieve relevant medical literature chunks."""
    query: str = Field(
        ...,
        description="Clinical query describing patient presentation or medical question",
        min_length=5,
        json_schema_extra={
            "examples": [
                "65-year-old male with bilateral lobar consolidation on chest X-ray, history of COPD"
            ]
        },
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "65-year-old male with bilateral lobar consolidation on chest X-ray, history of COPD"
                }
            ]
        }
    }


class QueryRequest(BaseModel):
    """Request for full RAG pipeline: retrieve + generate."""
    query: str = Field(
        ...,
        description="Clinical query describing patient presentation",
        min_length=5,
    )
    cnn_prediction: str = Field(
        default="unknown",
        description="CNN classification result (e.g., 'pneumonia', 'normal')",
    )
    confidence: float = Field(
        default=0.0,
        description="CNN prediction confidence (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "65-year-old male with bilateral lobar consolidation, history of COPD",
                    "cnn_prediction": "pneumonia",
                    "confidence": 0.87,
                }
            ]
        }
    }


# ── Response models ─────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single retrieved document chunk with metadata."""
    chunk_text: str = Field(description="Text content of the retrieved chunk")
    similarity: float = Field(description="Cosine similarity score to query")
    doc_id: str = Field(description="Source document identifier")
    chunk_index: int = Field(description="Chunk position within the source document")


class RetrieveResponse(BaseModel):
    """Response from the /retrieve endpoint."""
    query: str
    chunks: list[RetrievedChunk]
    retrieval_latency_ms: float = Field(description="Time spent on retrieval in milliseconds")
    total_latency_ms: float = Field(description="Total request time in milliseconds")


class QueryResponse(BaseModel):
    """Response from the /query endpoint (full RAG pipeline)."""
    query: str
    cnn_prediction: str
    confidence: float
    chunks: list[RetrievedChunk]
    generated_response: str = Field(description="LLM-generated clinical summary")
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    corpus_loaded: bool
    corpus_size: int
    chunk_count: int


class StatsResponse(BaseModel):
    """Response from the /stats endpoint."""
    corpus_size: int
    chunk_count: int
    embedding_model: str
    chunking_strategy: str
    retrieval_k: int
