# 🏥 Medical Imaging Clinical Decision Support System

> **End-to-end Clinical Decision Support System** — PyTorch CNN chest X-ray classification + RAG-powered medical literature retrieval. FastAPI backend, PostgreSQL, Docker, GCP deployment.

---

## ⚠️ Medical & Legal Disclaimer

**This is an educational demonstration project ONLY.**

- ❌ NOT intended for clinical or diagnostic use
- ❌ NOT a substitute for professional medical advice
- ❌ NOT FDA approved or clinically validated
- ❌ Do NOT use for actual medical diagnosis or treatment decisions
- ✅ For research, educational, and portfolio purposes only

All data used in this project is sourced from publicly available, legally accessible datasets. No private patient data (PHI) is used or stored at any point. See [Data Sources](#-data-sources--legal-compliance) for full details.

---

## 📋 Project Overview

This system combines **computer vision** and **retrieval-augmented generation (RAG)** to demonstrate a full-stack ML engineering pipeline applied to medical imaging.

**Pipeline:**
```
Chest X-ray Image
       ↓
CNN Classification (ResNet50 — PyTorch)
       ↓
RAG Retrieval (BioBERT embeddings → PostgreSQL vector search)
       ↓
LLM Clinical Summary with Citations
       ↓
FastAPI REST Endpoint → Docker → GCP Cloud Run
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, ResNet50 (transfer learning) |
| NLP / Embeddings | BioBERT, sentence-transformers |
| RAG | LangChain (architecture), custom retrieval pipeline |
| Vector Search | PostgreSQL + pgvector |
| Database | PostgreSQL (3 tables: documents, query logs, model versions) |
| Data Engineering | PubMed API (Biopython), ETL pipeline, cron scheduling |
| Backend API | FastAPI (async) |
| Containerization | Docker, Docker Compose |
| Cloud | GCP Cloud Run |
| Monitoring | Query logging, health checks, metrics endpoint |
| Evaluation | Precision@5, 20-case clinical test dataset, embedding A/B comparison |

---

## 📁 Project Structure

```
medical-imaging-clinical-support/
├── README.md
├── ARCHITECTURE.md              # Design decisions & tradeoff documentation
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── deploy.sh
│
├── database/
│   └── schema.sql               # PostgreSQL schema (3 tables)
│
├── pipelines/
│   ├── pubmed_etl.py            # PubMed API ingestion pipeline
│   └── scheduler.py             # Automated weekly updates
│
├── models/
│   └── cnn_trainer.py           # ResNet50 transfer learning pipeline
│
├── rag/
│   ├── embedding_pipeline.py    # BioBERT embedding generation
│   └── retrieval_pipeline.py    # RAG retrieval + LLM generation
│
├── api/
│   └── main.py                  # FastAPI application
│
├── evaluation/
│   ├── test_cases.json          # 20-case clinical evaluation set
│   ├── rag_evaluator.py         # Precision@5 measurement
│   └── results.md               # Documented comparison results
│
├── monitoring/
│   └── metrics.py               # System health and analytics
│
└── tests/
    ├── test_etl.py
    ├── test_rag.py
    └── test_api.py
```

---

## 🚀 Quickstart (Local)

### Prerequisites
- Python 3.10+
- PostgreSQL 15+
- Docker Desktop

### 1. Clone the repository
```bash
git clone https://github.com/HelloWorldfromhere/medical-imaging-clinical-support.git
cd medical-imaging-clinical-support
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Fill in: POSTGRES_PASSWORD, OPENAI_API_KEY
```

### 4. Run with Docker Compose
```bash
docker-compose up --build
```

### 5. Access the API
```
http://localhost:8000/docs   ← Swagger UI
http://localhost:8000/health ← Health check
```

---

## 📊 Evaluation Results

| Embedding Model | Precision@5 | Latency | Model Size |
|---|---|---|---|
| all-MiniLM-L6-v2 | ~68% | ~120ms | 80MB |
| BioBERT | ~82% | ~150ms | 420MB |

**Decision:** BioBERT selected — +14% precision justifies 30ms latency increase for medical accuracy.

Full comparison documented in [`evaluation/results.md`](evaluation/results.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 📚 Data Sources & Legal Compliance

All data used in this project is publicly available and legally accessible:

| Dataset | Source | License | Usage |
|---|---|---|---|
| ChestX-ray14 | NIH National Library of Medicine | Public Domain | CNN training |
| PubMed Abstracts | NLM E-Utilities API | Free public access (with attribution) | RAG knowledge base |

**PubMed API usage complies with:**
- NLM terms of service (email included in all requests)
- Rate limits respected (≤10 requests/second with API key)
- All abstracts cited with PMID

No private, identifiable, or HIPAA-protected patient data is used at any stage.

---

## 🎓 About

Built as a portfolio project demonstrating full-stack ML engineering skills.

**Author:** CS Student, Concordia University — Graduating May 2026
**Target Role:** Junior ML Engineer / Applied AI Engineer
**Contact:** https://www.linkedin.com/in/ionel-turcan-ab6890234/

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

This project is for educational and research purposes only. See [Medical Disclaimer](#️-medical--legal-disclaimer) above.
