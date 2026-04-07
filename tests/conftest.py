"""
Shared pytest fixtures — mock the RAG pipeline, CNN model, and app startup.
Tests run without a database, GPU, or loaded corpus.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake objects that replace heavy dependencies during testing
# ---------------------------------------------------------------------------

def _make_fake_doc(text="Pneumonia is treated with antibiotics...", sim=0.55, doc_id="pubmed_99999", idx=0):
    """Create a fake retrieved document matching what RAGPipeline.retrieve() returns."""
    return SimpleNamespace(chunk_text=text, similarity=sim, doc_id=doc_id, chunk_index=idx)


class FakePipeline:
    """Lightweight stand-in for RAGPipeline — no embeddings, no DB."""
    model_key = "minilm"
    strategy_key = "recursive_paragraph"
    k = 7
    _chunk_texts = ["chunk"] * 120  # pretend we have 120 chunks

    def build_index(self, documents):
        pass

    def retrieve(self, query):
        docs = [
            _make_fake_doc("Pneumonia community-acquired treatment...", 0.58, "pubmed_001", 0),
            _make_fake_doc("Chest X-ray consolidation findings...", 0.52, "pubmed_002", 1),
        ]
        return docs, 23.5  # (results, retrieval_ms)


FAKE_CNN_RESULT = {
    "predictions": [
        {"condition": "Pneumonia", "probability": 0.85, "detected": True},
        {"condition": "Consolidation", "probability": 0.62, "detected": True},
        {"condition": "Effusion", "probability": 0.15, "detected": False},
    ],
    "model_loaded": True,
    "needs_manual_selection": False,
}


# ---------------------------------------------------------------------------
# Fixture: patched FastAPI app + test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """
    Yield a TestClient whose app has all heavy dependencies mocked out.
    The patches replace:
      - RAG pipeline loading (no embedding model, no corpus file)
      - CNN model loading and prediction (no PyTorch weights)
      - LLM summary generation (no API key needed)
    """
    with (
        patch("api.main.load_pipeline"),
        patch("api.main.pipeline", new=FakePipeline()),
        patch("api.main.corpus_size", new=775),
        patch("api.main.chunk_count", new=120),
        patch("api.main.is_model_loaded", return_value=True),
        patch("api.main.predict_conditions", return_value=FAKE_CNN_RESULT),
        patch("api.main.generate_clinical_summary", return_value="This is a mock clinical summary for testing."),
    ):
        from api.main import app
        with TestClient(app) as tc:
            yield tc
