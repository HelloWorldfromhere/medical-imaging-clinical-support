# Medical Imaging Clinical Decision Support System

![CI](https://github.com/HelloWorldfromhere/medical-imaging-clinical-support/actions/workflows/ci.yml/badge.svg)

> **End-to-end Clinical Decision Support System** — PyTorch CNN chest X-ray classification + RAG-powered medical literature retrieval with hybrid BM25 + vector search. FastAPI backend, PostgreSQL, Docker, GCP deployment.

---

## Medical and Legal Disclaimer

**This is an educational demonstration project ONLY.**

- NOT intended for clinical or diagnostic use
- NOT a substitute for professional medical advice
- NOT FDA approved or clinically validated
- Do NOT use for actual medical diagnosis or treatment decisions
- For research, educational, and portfolio purposes only

All data used in this project is sourced from publicly available, legally accessible datasets. No private patient data (PHI) is used or stored at any point. See [Data Sources](#data-sources-and-legal-compliance) for full details.

---

## Project Overview

This system combines **computer vision** and **retrieval-augmented generation (RAG)** to demonstrate a full-stack ML engineering pipeline applied to medical imaging.

### Pipeline

```
Chest X-ray Image
       |
CNN Classification (EfficientNet-B3 -- PyTorch)
       |
Hybrid Retrieval (BM25 + MPNet vector search, RRF fusion)
       |
LLM Clinical Summary with Citations
       |
FastAPI REST API -- Docker -- GCP Cloud Run
```

### Key Differentiators

- **Systematic evaluation methodology:** 3 rounds of retrieval evaluation comparing 3 embedding models, 5 chunking strategies, and hybrid retrieval configurations — not just "it works," but quantified trade-offs documented in ARCHITECTURE.md
- **Hybrid retrieval with RRF fusion:** BM25 keyword matching + dense vector search combined via Reciprocal Rank Fusion, outperforming either method alone
- **Custom semantic evaluation metric:** Cosine similarity between retrieved chunks and expected topic descriptions, with topic coverage scoring across 20 clinical test cases
- **Production-ready API:** FastAPI backend with health checks, Pydantic validation, Docker containerization, and GCP Cloud Run deployment

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, EfficientNet-B3 (transfer learning) |
| NLP / Embeddings | all-mpnet-base-v2, all-MiniLM-L6-v2, BioLORD (compared) |
| Retrieval | Hybrid BM25 + vector search, Reciprocal Rank Fusion |
| Vector Search | PostgreSQL + pgvector, in-memory NumPy (dev) |
| Database | PostgreSQL (documents, query logs) |
| Data Engineering | PubMed E-Utilities API, ETL pipeline, JSON cache |
| Backend API | FastAPI (async), Pydantic v2 |
| Containerization | Docker, Docker Compose |
| Cloud | GCP Cloud Run |
| Evaluation | Semantic Precision, Topic Coverage, 20-case clinical test suite |

---

## Project Structure

```
medical-imaging-clinical-support/
├── README.md
├── ARCHITECTURE.md              # Design decisions and evaluation results
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application (4 endpoints)
│   └── schemas.py               # Pydantic request/response models
│
├── rag/
│   ├── __init__.py
│   ├── embedding_pipeline.py    # Embedding generation and comparison
│   └── retrieval_pipeline.py    # Hybrid retrieval + prompt + generation
│
├── models/
│   ├── cnn_trainer.py           # EfficientNet-B3 training pipeline
│   └── visualizer.py            # Training curves and confusion matrices
│
├── evaluation/
│   ├── __init__.py
│   ├── test_cases.json          # 20-case clinical evaluation set
│   ├── rag_evaluator.py         # Keyword-based evaluation (Round 1-2)
│   ├── semantic_evaluator.py    # Semantic evaluation (Round 3)
│   ├── results.json             # Detailed per-case scores
│   ├── results.md               # Comparison tables for ARCHITECTURE.md
│   └── plots/                   # CNN training curves, confusion matrices
│
├── pipelines/
│   ├── pubmed_etl.py            # PubMed API ingestion (PostgreSQL)
│   ├── pubmed_fetch_json.py     # PubMed fetch to JSON cache
│   └── pubmed_cache/
│       └── documents.json       # 775 PubMed abstracts (0 duplicates)
│
├── database/
│   └── schema.sql               # PostgreSQL schema
│
├── scripts/                     # Development and evaluation utilities
│   ├── test_chunking.py         # Chunking strategy comparison
│   ├── test_advanced_retrieval.py
│   ├── test_improvements.py
│   ├── test_spotcheck.py        # Visual chunk inspection
│   └── fetch_weak_conditions.py # Targeted corpus expansion
│
├── run_evaluation.py            # Full evaluation entry point
├── test_final_optimization.py   # Best config benchmark
└── test_api.py                  # API endpoint tests
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check (corpus loaded, chunk count) |
| GET | `/stats` | Corpus size, embedding model, retrieval config |
| POST | `/retrieve` | Retrieve top-k relevant chunks for a clinical query |
| POST | `/query` | Full RAG pipeline: retrieve + generate clinical summary |

### Example Request

```bash
curl -X POST http://localhost:8080/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "65-year-old male with bilateral lobar consolidation, history of COPD"}'
```

### Example Response

```json
{
  "query": "65-year-old male with bilateral lobar consolidation, history of COPD",
  "chunks": [
    {
      "chunk_text": "Community-acquired pneumonia in patients with COPD...",
      "similarity": 0.586,
      "doc_id": "pubmed_12345",
      "chunk_index": 3
    }
  ],
  "retrieval_latency_ms": 23.5,
  "total_latency_ms": 25.1
}
```

Interactive API documentation available at `/docs` (Swagger UI) when the server is running.

---

## Quickstart

### Prerequisites
- Python 3.10+
- PostgreSQL 15+ (optional — runs in-memory without it)
- Docker Desktop (for containerized deployment)

### Run Locally

```bash
git clone https://github.com/HelloWorldfromhere/medical-imaging-clinical-support.git
cd medical-imaging-clinical-support
pip install -r requirements.txt
uvicorn api.main:app --port 8080
```

Visit `http://localhost:8080/docs` for the interactive API playground.

### Run with Docker

```bash
docker compose up --build
```

Visit `http://localhost:8080/docs` — the container loads the embedding model and indexes 775 documents on startup (~40s).

### Run API Tests

```bash
# Start server in one terminal, then in another:
python test_api.py
```

---

## Evaluation Results

### CNN Model Comparison

| Model | Accuracy | Missed Pneumonia Cases | Decision |
|---|---|---|---|
| ResNet50 | Higher overall | More missed pneumonia | Rejected |
| EfficientNet-B3 | Competitive | 67% fewer missed pneumonia | **Selected** |

**Decision:** EfficientNet-B3 selected — fewer missed pneumonia cases is critical in medical imaging where false negatives have severe consequences.

### RAG Retrieval Evaluation (3 Rounds)

**Round 1-2:** Keyword-based evaluation across 3 embedding models and 3 chunking strategies. MiniLM outperformed BioBERT and PubMedBERT — BioBERT's BERT architecture was not trained for similarity tasks, producing poor retrieval despite domain specificity.

**Round 3:** Semantic evaluation with hybrid retrieval. Best configuration:

| Parameter | Value |
|---|---|
| Embedding Model | all-mpnet-base-v2 (768-dim) |
| Chunking | Recursive, 800 chars, 80 overlap |
| Retrieval | Hybrid BM25 (2x weight) + vector, RRF fusion |
| Candidates | 40 initial, relevance floor 0.30, final k=7 |

| Metric | Score |
|---|---|
| Semantic Precision | 0.562 |
| Topic Coverage | 76.7% |

**Key finding:** Cross-encoder reranking (negative result) degraded performance — documented as an honest experimental outcome in ARCHITECTURE.md.

Full methodology and decision rationale in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Corpus

- 775 PubMed abstracts (0 duplicates after content-hash deduplication)
- Built in 4 stages: 228 → 256 → 582 → 775
- Coverage verified for 15+ clinical conditions
- Sourced via PubMed E-Utilities API with dual-path ETL (PostgreSQL + JSON cache)

---

## Data Sources and Legal Compliance

All data used in this project is publicly available and legally accessible:

| Dataset | Source | License | Usage |
|---|---|---|---|
| Chest X-ray Dataset | NIH / Kaggle | Public Domain | CNN training and evaluation |
| PubMed Abstracts | NLM E-Utilities API | Free public access | RAG knowledge base (775 docs) |

PubMed API usage complies with NLM terms of service. Rate limits respected. All abstracts cited with PMID. No private, identifiable, or HIPAA-protected patient data is used at any stage.

---

## About

Built as a portfolio project demonstrating full-stack ML engineering: from model selection and systematic evaluation to production API deployment.

**Author:** Ion Turcan — CS Student, Concordia University (Graduating August 2026)

**Other work:** 3rd Place ($2,000 prize), Mila x Bell x Kids Help Phone AI Safety Hackathon (80+ teams) — built a multilingual content safety classifier with XLM-RoBERTa (F1=0.882)

**Contact:** [LinkedIn](https://www.linkedin.com/in/ionel-turcan-ab6890234/) | [GitHub](https://github.com/HelloWorldfromhere)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

This project is for educational and research purposes only. See [Medical Disclaimer](#medical-and-legal-disclaimer) above.
