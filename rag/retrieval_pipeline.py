"""
RAG Retrieval Pipeline — Query → Retrieve → Generate
Medical Imaging RAG Clinical Decision Support

Handles the full RAG flow:
1. Embed user query
2. Vector similarity search (pgvector or in-memory fallback)
3. Build prompt with retrieved context
4. Generate clinical summary via OpenAI API
5. Log query for monitoring

Usage:
    from rag.retrieval_pipeline import RAGPipeline

    pipeline = RAGPipeline(model_key="biobert", strategy_key="recursive_paragraph")
    response = pipeline.query(
        query="65-year-old male with bilateral consolidation, history of COPD",
        cnn_prediction="pneumonia",
        confidence=0.87
    )
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from rag.embedding_pipeline import EMBEDDING_MODELS, CHUNKING_STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    doc_id: str
    chunk_text: str
    similarity: float
    chunk_index: int


@dataclass
class RAGResponse:
    query: str
    cnn_prediction: str
    confidence: float
    retrieved_docs: list[RetrievedDocument]
    generated_response: str
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float


class RAGPipeline:
    """
    End-to-end RAG pipeline for medical imaging clinical decision support.

    Supports two retrieval backends:
    - PostgreSQL + pgvector (production)
    - In-memory numpy (development / evaluation)
    """

    def __init__(
        self,
        model_key: str = "biobert",
        strategy_key: str = "recursive_paragraph",
        k: int = 5,
        db_config: Optional[dict] = None,
        use_postgres: bool = False,
    ):
        self.model_key = model_key
        self.strategy_key = strategy_key
        self.k = k
        self.use_postgres = use_postgres
        self.db_config = db_config or {}

        model_info = EMBEDDING_MODELS[model_key]
        logger.info(f"Loading embedding model: {model_info['name']}")
        self.model = SentenceTransformer(model_info["name"])

        # In-memory index (populated by load_index or build_index)
        self._chunk_texts: list[str] = []
        self._chunk_doc_ids: list[str] = []
        self._chunk_indices: list[int] = []
        self._embeddings: Optional[np.ndarray] = None

    # ---- Index management --------------------------------------------------

    def build_index(self, documents: list[dict]):
        """Build in-memory index from raw documents (for dev/eval)."""
        from rag.embedding_pipeline import EmbeddingPipeline

        pipeline = EmbeddingPipeline()
        pipeline.documents = documents
        chunks = pipeline.chunk_documents(self.strategy_key)
        result = pipeline.embed_chunks(chunks, self.model_key)

        self._chunk_texts = [c.text for c in chunks]
        self._chunk_doc_ids = [c.doc_id for c in chunks]
        self._chunk_indices = [c.chunk_index for c in chunks]
        self._embeddings = result.embeddings

        logger.info(f"In-memory index built: {len(self._chunk_texts)} chunks")

    def load_index_from_file(self, embeddings_path: str, chunks_path: str):
        """Load pre-computed embeddings from numpy file + chunk metadata."""
        self._embeddings = np.load(embeddings_path)
        with open(chunks_path) as f:
            meta = json.load(f)
        self._chunk_texts = [m["text"] for m in meta]
        self._chunk_doc_ids = [m["doc_id"] for m in meta]
        self._chunk_indices = [m["chunk_index"] for m in meta]
        logger.info(f"Loaded index: {len(self._chunk_texts)} chunks from {embeddings_path}")

    def save_index(self, embeddings_path: str, chunks_path: str):
        """Save current index to disk for fast reload."""
        if self._embeddings is None:
            raise ValueError("No index to save — call build_index first")

        Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self._embeddings)

        meta = [
            {"text": t, "doc_id": d, "chunk_index": i}
            for t, d, i in zip(self._chunk_texts, self._chunk_doc_ids, self._chunk_indices)
        ]
        with open(chunks_path, "w") as f:
            json.dump(meta, f)
        logger.info(f"Index saved: {embeddings_path}, {chunks_path}")

    # ---- Retrieval ---------------------------------------------------------

    def retrieve(self, query: str) -> tuple[list[RetrievedDocument], float]:
        """
        Retrieve top-k relevant chunks for a query.
        Returns (documents, latency_ms).
        """
        start = time.perf_counter()

        if self.use_postgres:
            docs = self._retrieve_postgres(query)
        else:
            docs = self._retrieve_memory(query)

        latency = (time.perf_counter() - start) * 1000
        return docs, round(latency, 2)

    def _retrieve_memory(self, query: str) -> list[RetrievedDocument]:
        """In-memory cosine similarity search."""
        if self._embeddings is None:
            raise ValueError("No index loaded — call build_index or load_index_from_file")

        query_emb = self.model.encode([query])
        sims = cosine_similarity(query_emb, self._embeddings)[0]
        top_indices = np.argsort(sims)[-self.k:][::-1]

        return [
            RetrievedDocument(
                doc_id=self._chunk_doc_ids[i],
                chunk_text=self._chunk_texts[i],
                similarity=float(sims[i]),
                chunk_index=self._chunk_indices[i],
            )
            for i in top_indices
        ]

    def _retrieve_postgres(self, query: str) -> list[RetrievedDocument]:
        """PostgreSQL pgvector similarity search."""
        query_emb = self.model.encode([query]).tolist()[0]

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT doc_id, chunk_text, chunk_index,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_embeddings
            WHERE embedding_model = %s AND chunking_strategy = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_emb, self.model_key, self.strategy_key, query_emb, self.k),
        )

        docs = [
            RetrievedDocument(
                doc_id=row[0],
                chunk_text=row[1],
                chunk_index=row[2],
                similarity=float(row[3]),
            )
            for row in cur.fetchall()
        ]

        cur.close()
        conn.close()
        return docs

    # ---- Prompt construction -----------------------------------------------

    def build_prompt(
        self,
        query: str,
        cnn_prediction: str,
        confidence: float,
        retrieved_docs: list[RetrievedDocument],
    ) -> str:
        """Build the LLM prompt with retrieved context and CNN results."""
        context_block = "\n\n".join(
            f"[Source {i+1}: {doc.doc_id}] (relevance: {doc.similarity:.3f})\n{doc.chunk_text}"
            for i, doc in enumerate(retrieved_docs)
        )

        return f"""You are a medical AI assistant providing clinical decision support.
A chest X-ray has been analyzed by an AI classification model.

**AI Classification Result:**
- Prediction: {cnn_prediction}
- Confidence: {confidence:.1%}

**Patient Presentation:**
{query}

**Retrieved Medical Literature:**
{context_block}

**Instructions:**
Based on the above classification and retrieved literature, provide:
1. Clinical interpretation of the AI finding
2. Differential diagnoses to consider
3. Recommended next diagnostic steps
4. Relevant treatment considerations
5. Any limitations or uncertainties

Cite sources using [Source N] format. Include a disclaimer that this is AI-generated
and requires physician review.

**Clinical Summary:**"""

    # ---- Generation (stubbed for OpenAI / local LLM) -----------------------

    def generate(self, prompt: str) -> tuple[str, float]:
        """
        Generate clinical summary from prompt.
        Replace with actual LLM call (OpenAI, local model, etc.)
        Returns (response_text, latency_ms).
        """
        start = time.perf_counter()

        # --- STUB: Replace with actual LLM call ---
        # Option A: OpenAI
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are a medical AI assistant."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.3,
        #     max_tokens=500
        # )
        # text = response.choices[0].message.content

        # Option B: Stub response for evaluation without API key
        text = (
            "[STUB] This is a placeholder response. To enable LLM generation, "
            "configure OpenAI API key in .env and uncomment the generation code "
            "in rag/retrieval_pipeline.py.\n\n"
            "The retrieval component is functional and can be evaluated independently."
        )

        latency = (time.perf_counter() - start) * 1000
        return text, round(latency, 2)

    # ---- Full query pipeline -----------------------------------------------

    def query(
        self,
        query: str,
        cnn_prediction: str = "unknown",
        confidence: float = 0.0,
    ) -> RAGResponse:
        """Execute the full RAG pipeline: retrieve → prompt → generate."""
        total_start = time.perf_counter()

        # Step 1: Retrieve
        retrieved_docs, retrieval_latency = self.retrieve(query)

        # Step 2: Build prompt
        prompt = self.build_prompt(query, cnn_prediction, confidence, retrieved_docs)

        # Step 3: Generate
        generated_text, generation_latency = self.generate(prompt)

        total_latency = (time.perf_counter() - total_start) * 1000

        response = RAGResponse(
            query=query,
            cnn_prediction=cnn_prediction,
            confidence=confidence,
            retrieved_docs=retrieved_docs,
            generated_response=generated_text,
            total_latency_ms=round(total_latency, 2),
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
        )

        logger.info(
            f"Query completed: {len(retrieved_docs)} docs retrieved in "
            f"{retrieval_latency:.1f}ms, total {total_latency:.1f}ms"
        )
        return response

    # ---- Query logging -----------------------------------------------------

    def log_query(self, response: RAGResponse, db_config: Optional[dict] = None):
        """Log query to PostgreSQL for monitoring and analytics."""
        config = db_config or self.db_config
        if not config:
            logger.warning("No DB config — skipping query log")
            return

        conn = psycopg2.connect(**config)
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO query_logs
                (query_text, prediction, confidence, retrieved_doc_ids,
                 response_text, latency_ms, success)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                response.query,
                response.cnn_prediction,
                response.confidence,
                json.dumps([d.doc_id for d in response.retrieved_docs]),
                response.generated_response[:2000],
                response.total_latency_ms,
                True,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
