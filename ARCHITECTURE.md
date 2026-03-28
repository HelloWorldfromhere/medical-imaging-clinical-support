# Architecture Documentation
## Medical Imaging + RAG Clinical Decision Support System

> **Purpose:** This document records every significant architectural decision made during development — what was chosen, what alternatives were considered, and *why*. This is the difference between a portfolio project and a tutorial copy.

---

## Table of Contents

1. [Project Phases](#1-project-phases)
2. [Dataset Decisions](#2-dataset-decisions)
3. [CNN Architecture](#3-cnn-architecture)
4. [Training Configuration](#4-training-configuration)
5. [Model Evaluation & Selection](#5-model-evaluation--selection)
6. [Database Design](#6-database-design)
7. [ETL Pipeline Design](#7-etl-pipeline-design)
8. [RAG System Design — Phase 3](#8-rag-system-design--phase-3)
9. [Deployment Architecture (Planned — Phase 4)](#9-deployment-architecture-planned--phase-4)
10. [Key Learnings & Trade-offs](#10-key-learnings--trade-offs)

---

## 1. Project Phases

The system is built in deliberate phases to allow early validation before committing to full complexity.

| Phase | Scope | Status | Rationale |
|-------|-------|--------|-----------|
| **Phase 1** | Binary CNN classifier (NORMAL vs PNEUMONIA) | ✅ Complete | Establish baseline, validate training pipeline, prove clinical metric reasoning |
| **Phase 2** | Multi-label CNN classifier (ChestX-ray14, 14 conditions) | ⬜ Week 5 | Real clinical utility requires multi-condition detection |
| **Phase 3** | RAG pipeline — PubMed literature retrieval per detected condition | ✅ Evaluation Complete | LLM-powered clinical context for each detected pathology |
| **Phase 4** | FastAPI backend + Docker + GCP Cloud Run deployment | ⬜ Week 7 | Production-grade, publicly accessible system |

**Why phase it this way?**
Starting with binary classification allows validation of the entire training pipeline (data loading, augmentation, GPU training, evaluation, database logging) on a well-understood problem before scaling to 14-class multi-label classification. A common mistake is attempting full complexity before the infrastructure is proven.

---

## 2. Dataset Decisions

### Phase 1: Kaggle Chest X-Ray Images (Pneumonia)

**Dataset:** Paul Mooney / Guangzhou Women and Children's Medical Center
**Source:** https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
**Size:** 5,856 images | 2 classes: NORMAL, PNEUMONIA
**Split:** 5,216 train / 16 val / 624 test

**Decision: Use this dataset first**

| Option | Size | Classes | Pros | Cons |
|--------|------|---------|------|------|
| Kaggle Chest X-Ray | 5,856 | 2 | Fast iteration, clean labels, well-validated | Binary only, limited clinical utility |
| **ChestX-ray14 (NIH)** | **112,120** | **14** | **Production-realistic, multi-label** | **45GB, complex architecture, longer training** |
| MIMIC-CXR | 227,827 | 14+ | Largest dataset, radiology reports | Requires credentialed access |

**Rationale:** Phase 1 uses the Kaggle dataset to prove the training pipeline works end-to-end. The small validation set (16 images) causes val accuracy oscillation — this is a known dataset artifact, not a model failure. Phase 2 migrates to ChestX-ray14 for real clinical utility.

**Data Augmentation Strategy:**
```python
# Training transforms — chosen for medical imaging constraints
transforms.RandomHorizontalFlip()          # Anatomically valid for chest X-rays
transforms.RandomRotation(10)              # Small rotations only — preserve orientation
transforms.ColorJitter(brightness=0.2,     # Simulate acquisition variability
                       contrast=0.2)
transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                     std=[0.229, 0.224, 0.225])     # Valid for transfer learning
```

**Why not RandomVerticalFlip?**
Vertical flipping would invert lung anatomy (apex/base orientation), creating anatomically impossible images that could confuse the model on real data.

---

## 3. CNN Architecture

### Decision: ResNet50 vs EfficientNet-B3 (Comparison Study)

Rather than selecting one architecture arbitrarily, both were trained and evaluated. This comparison is the core of Phase 1's portfolio value.

| Criterion | ResNet50 | EfficientNet-B3 |
|-----------|----------|-----------------|
| Parameters | 25.6M | 12.3M |
| ImageNet Top-1 | 76.1% | 81.6% |
| Pre-trained weights | torchvision | torchvision |
| Training time (15 epochs, RTX 3060) | ~18.5 min | ~18.5 min |
| Model size on disk | ~98MB | ~48MB |

**Why these two specifically?**

- **ResNet50** is the industry standard baseline for medical imaging transfer learning. It appears in the majority of published chest X-ray classification papers, making it the right comparison anchor.
- **EfficientNet-B3** represents the compound-scaling approach — systematically balancing depth, width, and resolution. Smaller parameter count with higher ImageNet accuracy suggests better feature efficiency.

**Transfer Learning Strategy:**
```python
# Frozen feature extractor + replaced classifier head
# ResNet50
model.fc = nn.Linear(2048, 2)

# EfficientNet-B3
model.classifier[1] = nn.Linear(1536, 2)

# Why freeze early layers?
# ImageNet low-level features (edges, textures) transfer directly to X-rays.
# Only the classification head needs domain-specific learning.
# Reduces training time 60-70% vs training from scratch.
# Prevents overfitting on small medical datasets.
```

**Why not VGG16, DenseNet121, or Vision Transformer?**

- **VGG16:** 138M parameters — excessive for a binary classifier on 5K images, high overfitting risk
- **DenseNet121:** Strong medical imaging performer (used in CheXNet paper), planned for Phase 2 comparison
- **ViT (Vision Transformer):** Requires much larger datasets (100K+) to outperform CNNs — better suited for ChestX-ray14 Phase 2

---

## 4. Training Configuration

```python
# Hyperparameters — with rationale

BATCH_SIZE = 32
# 32 chosen to fit RTX 3060 6GB VRAM with room for gradient computation.
# 64 caused OOM errors. 16 underutilized GPU. 32 is the stable midpoint.

LEARNING_RATE = 0.001
# Standard Adam LR for fine-tuning pre-trained models.
# Higher (0.01) caused validation loss instability in early experiments.
# Lower (0.0001) slowed convergence without accuracy benefit.

NUM_EPOCHS = 15
# Sufficient for convergence on 5K images with pre-trained weights.
# Training curves show both models converge by epoch 10-13.
# Early stopping triggered at best val_acc checkpoint.

OPTIMIZER = Adam
# Adam chosen over SGD for faster convergence on fine-tuning tasks.
# SGD with momentum requires more careful LR scheduling.

LOSS = CrossEntropyLoss
# Standard for multi-class classification.
# Note: Phase 2 will switch to BCEWithLogitsLoss for multi-label.

CLASS_WEIGHTS = computed from training distribution
# NORMAL: 1341 images, PNEUMONIA: 3875 images
# Class imbalance ratio ~1:2.9 — weights applied to loss function
# to prevent model from always predicting PNEUMONIA.
```

---

## 5. Model Evaluation & Selection

### Results Summary

| Metric | ResNet50 | EfficientNet-B3 | Clinical Implication |
|--------|----------|-----------------|----------------------|
| Test Accuracy | **85.90%** | 85.42% | Overall correctness |
| Pneumonia Recall | 97.69% | **99.23%** | % of sick patients correctly identified |
| Normal Recall | 66.24% | 62.39% | % of healthy patients correctly cleared |
| Pneumonia F1 | **0.896** | 0.895 | Harmonic mean for positive class |
| Normal F1 | **0.779** | 0.762 | Harmonic mean for negative class |
| False Negatives | 9 | **3** | Missed pneumonia cases (most critical) |
| False Positives | 79 | 88 | Healthy patients flagged for follow-up |

### Why EfficientNet-B3 Was Selected for Production

**Test accuracy alone is a misleading metric for medical classifiers.**

The confusion matrices reveal the critical difference:

- **ResNet50** missed 9 pneumonia cases (false negatives)
- **EfficientNet-B3** missed 3 pneumonia cases (false negatives) — a **67% reduction**

In clinical screening:
- **False Negative (missed pneumonia):** Patient sent home untreated → potential rapid deterioration, sepsis, death
- **False Positive (healthy patient flagged):** Patient receives additional screening → minor inconvenience, no harm

EfficientNet-B3 has 9 more false positives (88 vs 79), meaning slightly more healthy patients are flagged for follow-up. This is the **correct clinical trade-off**: err on the side of over-detection, not under-detection.

**EfficientNet-B3 is designated `is_active = TRUE` in the model_versions table.**

### Evaluation Plots

All plots saved to `evaluation/plots/`:

| Plot | Purpose |
|------|---------|
| `resnet50_training_curves.png` | Loss/accuracy over epochs — shows convergence |
| `resnet50_confusion_matrix.png` | Per-class prediction breakdown |
| `resnet50_sample_predictions.png` | Visual inspection of predictions |
| `efficientnet_b3_training_curves.png` | Loss/accuracy over epochs |
| `efficientnet_b3_confusion_matrix.png` | Per-class prediction breakdown — shows 3 vs 9 FN |
| `efficientnet_b3_sample_predictions.png` | Visual inspection |
| `model_comparison.png` | Side-by-side bar chart — all metrics |

---

## 6. Database Design

### Decision: PostgreSQL over SQLite or MongoDB

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **PostgreSQL** | JSONB support, pgvector extension for embeddings (Phase 3), ACID compliance, production standard | Requires server setup | **Selected** |
| SQLite | Zero setup, file-based | No vector search, not production-scalable | Rejected |
| MongoDB | Flexible schema | No native vector search without Atlas, weaker SQL | Rejected |
| Pinecone only | Purpose-built vector DB | No relational data storage | Phase 3 supplement |

**Why PostgreSQL specifically?**
Phase 3 requires vector similarity search for RAG embeddings. PostgreSQL + pgvector extension provides this natively, eliminating the need for a separate vector database. One database handles relational data (documents, logs, model versions) and vector search.

### Schema Decisions

```sql
-- medical_documents: JSONB for metadata
-- Rationale: PubMed records have variable fields (some have MeSH terms,
-- some have DOIs, some have author affiliations). JSONB avoids 30+ nullable columns.

-- model_versions: metrics as JSONB
-- Rationale: Different models report different metrics. JSONB allows storing
-- any metric set without schema migrations as the project evolves.

-- query_logs: tracks every RAG query
-- Rationale: Enables analysis of which clinical queries are most common,
-- which retrieved documents were useful, and where the system fails.
```

---

## 7. ETL Pipeline Design

### PubMed ETL (`pipelines/pubmed_etl.py`)

**Decision: Biopython Entrez over direct HTTP requests**

Biopython's Entrez module handles NCBI's rate limiting (3 req/sec without API key, 10/sec with), retry logic, and XML parsing. Direct HTTP requests would require reimplementing all of this.

**Search Terms Selected:**
```python
search_terms = [
    "chest X-ray pneumonia classification deep learning",
    "medical imaging convolutional neural network",
    "clinical decision support radiology AI",
    "chest radiograph automated diagnosis",
    "pulmonary disease machine learning diagnosis"
]
# 50 results per term x 5 terms = 250 max, 228 after deduplication
```

**Deduplication Strategy:**
```sql
ON CONFLICT (pubmed_id) DO NOTHING
-- PubMed IDs are globally unique. Any duplicate fetch is safely ignored.
-- This makes the ETL pipeline idempotent — safe to run multiple times.
```

**Why idempotency matters:**
The scheduler runs weekly. If a network error causes a partial run, re-running must not create duplicate records or raise errors. `ON CONFLICT DO NOTHING` guarantees this.

### Dual ETL Paths

The system provides two data acquisition paths for different environments:

| Path | File | Output | Use Case |
|------|------|--------|----------|
| **Production** | `pipelines/pubmed_etl.py` | PostgreSQL `medical_documents` table | Deployed system with database |
| **Development** | `pipelines/pubmed_fetch_json.py` | `pipelines/pubmed_cache/documents.json` | Local evaluation without database |

Both paths use the same PubMed API, same parsing logic, and same deduplication. The JSON path was added during Phase 3 evaluation to enable rapid iteration on embedding model comparisons without requiring a running PostgreSQL instance.

### Corpus Composition

The medical document corpus was built in two stages to ensure coverage of all clinical scenarios in the evaluation test dataset:

| Stage | Documents | Search Terms | Conditions Covered |
|-------|-----------|--------------|-------------------|
| Initial ETL | 228 | 5 chest radiology terms | Pneumonia, pneumothorax, effusion, infiltrates |
| Targeted expansion | 28 | 3 gap-filling terms | Lupus pleuritis, heart failure, cardiomegaly |
| **Total** | **256** | **8 unique terms** | **13 clinical conditions** |

**Gap analysis approach:** After building the initial corpus, a keyword coverage check identified conditions with fewer than 3 mentions (lupus: 1, heart failure: 2). Targeted PubMed queries filled these gaps without re-fetching existing content. This approach respects PubMed's rate limits and demonstrates data-driven corpus design.

### Scheduler (`pipelines/scheduler.py`)

**Decision: `schedule` library now, Apache Airflow in Phase 4**

| Tool | Setup time | Production-ready | Monitoring | Decision |
|------|-----------|-----------------|------------|----------|
| `schedule` | 5 minutes | No | None | **Phase 1-3 (now)** |
| Apache Airflow | 2-4 hours | Yes | Full DAG UI | **Phase 4 migration** |
| Cron job | 10 minutes | Partial | None | Considered, rejected |

**Rationale:** The `schedule` library is sufficient for weekly ETL updates during development. The migration path to Airflow is documented and planned for Week 7 when the full deployment stack is built. Starting with Airflow would add infrastructure complexity before the core ML system is validated.

---

## 8. RAG System Design — Phase 3

### Overview

The RAG (Retrieval-Augmented Generation) pipeline retrieves relevant medical literature based on CNN classification results and generates clinical decision support summaries. Phase 3 focused on systematic evaluation of the retrieval component — selecting the optimal embedding model and chunking strategy through controlled experimentation.

### Evaluation Methodology

**Test Dataset:** 20 clinical scenarios covering 7 pathology categories (pneumonia variants, pleural conditions, cardiac findings, pneumothorax, pulmonary fibrosis, tuberculosis, and normal X-ray differentials). Each test case includes a clinical query, expected CNN prediction, expected medical keywords, and difficulty classification (standard vs complex).

**Metrics:**
- **Precision@5:** Fraction of top-5 retrieved chunks containing at least one expected keyword from the test case
- **Keyword Coverage:** Fraction of expected medical terms found across all top-5 retrieved chunks
- **Latency:** Time from query embedding to ranked results (milliseconds)

**Corpus:** 256 real PubMed abstracts fetched via Biopython Entrez API, covering 13 clinical conditions. All documents are publicly available medical literature cited by PMID.

### Round 1: Initial Model Selection (Base BERT Models)

The initial evaluation compared models selected based on domain relevance:

| Rank | Model | Strategy | P@5 | KW-Cov | Latency |
|------|-------|----------|-----|--------|---------|
| 1 | all-MiniLM-L6-v2 | fixed_512 | **0.290** | 0.200 | 7.1ms |
| 2 | all-MiniLM-L6-v2 | recursive | 0.240 | 0.140 | 6.9ms |
| 3 | all-MiniLM-L6-v2 | sentence | 0.230 | 0.152 | 11.7ms |
| 4 | BioBERT | fixed_512 | 0.120 | 0.120 | 10.4ms |
| 5 | PubMedBERT | fixed_512 | 0.100 | 0.120 | 12.4ms |
| 6-9 | BioBERT/PubMedBERT | recursive/sentence | 0.050-0.090 | 0.062-0.090 | 14-19ms |

**Unexpected result:** The general-purpose model (MiniLM) outperformed both biomedical models by a factor of 2-3x across all chunking strategies.

**Root cause analysis:** The sentence-transformers library logged `"No sentence-transformers model found with name dmis-lab/biobert-base-cased-v1.2. Creating a new one with mean pooling."` BioBERT and PubMedBERT are base BERT models trained for token-level tasks (named entity recognition, text classification) — not for producing sentence-level similarity embeddings. When loaded as sentence transformers, they default to naive mean pooling of token embeddings, which produces low-quality sentence representations. MiniLM, by contrast, was trained with contrastive learning specifically to produce embeddings where semantically similar sentences are close in vector space — exactly what retrieval requires.

**Key insight:** Domain specificity in pre-training does not guarantee retrieval quality. The training objective (contrastive sentence similarity vs masked language modeling) matters more than the training corpus for embedding-based retrieval tasks.

### Round 2: Corrected Model Selection (Sentence Transformer Models)

Based on the Round 1 diagnosis, biomedical models trained specifically as sentence transformers were selected:

| Rank | Model | Strategy | P@5 | KW-Cov | Latency |
|------|-------|----------|-----|--------|---------|
| 1 | all-MiniLM-L6-v2 | fixed_512 | **0.290** | **0.200** | 14.5ms |
| 2 | BioLORD-2023-M | fixed_512 | 0.280 | 0.172 | 10.4ms |
| 3 | BioLORD-2023-M | recursive | 0.280 | 0.152 | 12.2ms |
| 4 | S-PubMedBERT-MS-MARCO | fixed_512 | 0.270 | 0.182 | 11.1ms |
| 5 | S-PubMedBERT-MS-MARCO | recursive | 0.250 | 0.193 | 56.8ms |
| 6 | all-MiniLM-L6-v2 | recursive | 0.240 | 0.140 | 11.6ms |
| 7 | all-MiniLM-L6-v2 | sentence | 0.230 | 0.152 | 9.5ms |
| 8 | BioLORD-2023-M | sentence | 0.200 | 0.142 | 77.0ms |
| 9 | S-PubMedBERT-MS-MARCO | sentence | 0.160 | 0.113 | 72.7ms |

**Model descriptions:**
- **all-MiniLM-L6-v2:** General-purpose sentence transformer, 80 MB, 384-dimensional embeddings. Trained with contrastive learning on 1B+ sentence pairs.
- **BioLORD-2023-M:** Biomedical sentence transformer, 420 MB, 768-dimensional embeddings. Trained on biomedical concept relationships from UMLS and clinical ontologies.
- **S-PubMedBERT-MS-MARCO:** PubMedBERT fine-tuned on MS MARCO passage retrieval dataset, 420 MB, 768-dimensional embeddings. Combines biomedical language understanding with search-optimized similarity.

### Analysis of Results

**Model comparison:** With proper sentence transformer models, the biomedical models (BioLORD P@5=0.280, S-PubMedBERT P@5=0.270) closed the gap with MiniLM (P@5=0.290). The 3.4-7% difference is within noise range for 20 test cases, suggesting that for this corpus size and query complexity, the training objective (sentence similarity) matters more than domain-specific pre-training.

**Chunking strategy comparison:** Fixed 512-character chunks consistently outperformed recursive and sentence-based splitting across all models. This was unexpected — the hypothesis was that paragraph-aware recursive splitting would preserve clinical context better. The likely explanation: PubMed abstracts are already structured as coherent paragraphs, so fixed-size splitting at 512 characters rarely breaks mid-thought. Recursive splitting at 450 characters with 50-character overlap created more chunks (1,332 vs 888) with smaller average size (293 vs 437 chars), diluting the semantic signal per chunk.

**Sentence-based chunking performed worst** across all models. At 160-character average chunk size and 2,426 total chunks, individual chunks lacked sufficient context for meaningful similarity matching. A sentence like "Empiric antibiotics should be initiated" is too short to differentiate between pneumonia, UTI, or wound infection contexts.

### Selected Configuration

**Model:** all-MiniLM-L6-v2
**Chunking:** Fixed 512-character chunks, no overlap
**k:** 5 documents per retrieval

**Rationale:** MiniLM achieved the highest Precision@5 (0.290) and keyword coverage (0.200) while being the smallest model (80 MB vs 420 MB), fastest to load, and fastest at inference. The biomedical models provided no statistically significant improvement on this corpus to justify their 5x size increase.

**Production consideration:** For a larger corpus (1,000+ documents) or more specialized clinical queries, BioLORD may provide a meaningful advantage due to its biomedical ontology training. The evaluation framework (`run_evaluation.py`) supports re-running the comparison as the corpus grows.

### Evaluation Metric Limitations

The current Precision@5 metric uses exact keyword matching, which underestimates retrieval quality. A chunk discussing "trimethoprim-sulfamethoxazole" would not match the expected keyword "TMP-SMX" despite referring to the same drug. Similarly, a chunk about "antibiotic therapy for community-acquired pneumonia" is clinically relevant to a pneumonia query but scores zero if it does not contain the specific expected keywords like "consolidation" or "sputum culture."

A semantic similarity-based evaluation metric (comparing embedding similarity between retrieved chunks and expected topics rather than keyword presence) is planned as a refinement. The current keyword-based metric provides a conservative lower bound on retrieval quality.

### RAG Pipeline Architecture

```
Clinical Query + CNN Prediction
        |
        v
  Query Embedding (MiniLM, 384-dim)
        |
        v
  Vector Similarity Search
  (cosine similarity, top-5)
        |
        v
  Retrieved Chunks (5 x ~512 chars)
        |
        v
  Prompt Construction
  (query + CNN result + retrieved context)
        |
        v
  LLM Generation (OpenAI GPT-4, temp=0.3)
        |
        v
  Clinical Summary with Citations
        |
        v
  Query Logged to PostgreSQL
```

**Two retrieval backends:**
- **In-memory (numpy):** Used during development and evaluation. All chunk embeddings stored in a numpy array, cosine similarity computed with scikit-learn.
- **PostgreSQL + pgvector:** Production path. Embeddings stored as `vector` type columns, similarity search via `<=>` operator with automatic indexing.

---

## 9. Deployment Architecture (Planned — Phase 4)

```
User Query
    |
    v
Streamlit/React Frontend
    |
    v
FastAPI Backend (Cloud Run)
    |-- CNN Inference (EfficientNet-B3)
    |     |
    |   Detected Conditions
    |     |
    '-- RAG Pipeline
          |-- MiniLM Embedding (384-dim)
          |-- PostgreSQL pgvector Search (k=5)
          |-- Document Metadata Fetch
          '-- LLM Response Generation
                |
          Clinical Report to User
```

**Why GCP Cloud Run over AWS Lambda or Azure Functions?**

- Cloud Run supports Docker containers natively — no vendor-specific packaging
- PyTorch inference models are too large for Lambda's 250MB limit
- GCP free tier includes 2M requests/month — sufficient for portfolio demo
- Cloud Run scales to zero when not in use — no idle costs

**Why FastAPI over Flask?**

- Native async support for concurrent inference requests
- Automatic OpenAPI docs generation (`/docs`) — professional API presentation
- Pydantic validation — enforces input schemas for clinical data (important for regulated domain)
- 2-3x faster than Flask for I/O-bound tasks

---

## 10. Key Learnings & Trade-offs

| Decision | Trade-off | What We Sacrificed | What We Gained |
|----------|-----------|-------------------|----------------|
| EfficientNet-B3 over ResNet50 | Lower overall accuracy | 0.48% test accuracy | 67% fewer missed pneumonia cases |
| PostgreSQL over SQLite | Setup complexity | 30 min setup time | Production scalability, vector search |
| `schedule` over Airflow | No DAG monitoring | Visual pipeline UI | 2 hours saved, documented migration path |
| MiniLM over BioLORD | No domain-specific pre-training | Biomedical ontology awareness | 5x smaller model, faster inference, equal or better P@5 |
| Fixed 512 over recursive chunking | No paragraph-aware splitting | Context-preserving boundaries | Higher retrieval precision on PubMed abstracts |
| Sentence transformers over base BERT | Fewer model options | Access to raw BioBERT/PubMedBERT | 2-3x higher retrieval precision through proper training objective |
| Phase 1 binary before Phase 2 multi-label | Delayed clinical utility | 4 weeks | Proven infrastructure, clean baseline metrics |
| GCP Cloud Run over bare VM | Less control | SSH access | Auto-scaling, zero idle cost, managed SSL |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-06 | Initial ARCHITECTURE.md — Phase 1 complete | Ion Turcan |
| 2026-03-27 | Phase 3 RAG evaluation — 2 rounds, 9 configurations each | Ion Turcan |
| — | Phase 2 results (ChestX-ray14) | TBD Week 5 |
| — | Phase 4 deployment documentation | TBD Week 7 |
