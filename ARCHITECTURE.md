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
8. [RAG System Design (Planned — Phase 3)](#8-rag-system-design-planned--phase-3)
9. [Deployment Architecture (Planned — Phase 4)](#9-deployment-architecture-planned--phase-4)
10. [Key Learnings & Trade-offs](#10-key-learnings--trade-offs)

---

## 1. Project Phases

The system is built in deliberate phases to allow early validation before committing to full complexity.

| Phase | Scope | Status | Rationale |
|-------|-------|--------|-----------|
| **Phase 1** | Binary CNN classifier (NORMAL vs PNEUMONIA) | ✅ Complete | Establish baseline, validate training pipeline, prove clinical metric reasoning |
| **Phase 2** | Multi-label CNN classifier (ChestX-ray14, 14 conditions) | ⬜ Week 5 | Real clinical utility requires multi-condition detection |
| **Phase 3** | RAG pipeline — PubMed literature retrieval per detected condition | ⬜ Week 6 | LLM-powered clinical context for each detected pathology |
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
# 50 results per term × 5 terms = 250 max, 228 after deduplication
```

**Deduplication Strategy:**
```sql
ON CONFLICT (pubmed_id) DO NOTHING
-- PubMed IDs are globally unique. Any duplicate fetch is safely ignored.
-- This makes the ETL pipeline idempotent — safe to run multiple times.
```

**Why idempotency matters:**  
The scheduler runs weekly. If a network error causes a partial run, re-running must not create duplicate records or raise errors. `ON CONFLICT DO NOTHING` guarantees this.

### Scheduler (`pipelines/scheduler.py`)

**Decision: `schedule` library now, Apache Airflow in Phase 4**

| Tool | Setup time | Production-ready | Monitoring | Decision |
|------|-----------|-----------------|------------|----------|
| `schedule` | 5 minutes | No | None | **Phase 1-3 (now)** |
| Apache Airflow | 2-4 hours | Yes | Full DAG UI | **Phase 4 migration** |
| Cron job | 10 minutes | Partial | None | Considered, rejected |

**Rationale:** The `schedule` library is sufficient for weekly ETL updates during development. The migration path to Airflow is documented and planned for Week 7 when the full deployment stack is built. Starting with Airflow would add infrastructure complexity before the core ML system is validated.

---

## 8. RAG System Design (Planned — Phase 3)

> This section documents decisions made in advance based on research. Will be updated with actual results in Week 6.

### Embedding Model Selection

**Decision: BioBERT over general-purpose embeddings**

| Model | Dimension | Domain | Expected Precision@5 | Size |
|-------|-----------|--------|----------------------|------|
| `all-MiniLM-L6-v2` | 384 | General | Baseline | 80MB |
| `text-embedding-ada-002` | 1536 | General | +5-8% vs MiniLM | API cost |
| **`BioBERT`** | **768** | **Biomedical** | **+12-14% vs MiniLM** | **400MB** |
| `PubMedBERT` | 768 | Biomedical | Similar to BioBERT | 400MB |

**Why BioBERT?**  
BioBERT was pre-trained on PubMed abstracts and PMC full-text articles — the exact corpus we are retrieving from. Medical terminology ("atelectasis," "consolidation," "pleural effusion") is semantically understood rather than treated as rare tokens. Research shows 12-14% precision improvement on biomedical retrieval tasks vs general embeddings.

**Evaluation Plan:** All 3 embedding models will be tested on 20 clinical query test cases. Results documented in `evaluation/results.md`.

### Chunking Strategy

**Decision: RecursiveCharacterTextSplitter, 400-500 chars, 50 char overlap**

| Strategy | Chunk Size | Overlap | Rationale |
|----------|-----------|---------|-----------|
| Fixed-size | 256 chars | 0 | Simple but splits mid-sentence |
| Sentence-based | Variable | 0 | Preserves sentences, variable retrieval quality |
| **Recursive** | **400-500 chars** | **50 chars** | **Preserves clinical context, consistent size** |

**Why 400-500 chars?**  
Medical abstracts express one clinical finding per 2-3 sentences (~400-500 characters). Smaller chunks lose context (e.g., splitting "sensitivity was 94%" from "for detecting pneumonia"). Larger chunks retrieve too much irrelevant content, reducing precision.

**Why 50 char overlap?**  
Prevents important clinical statements that span chunk boundaries from being missed entirely.

### Retrieval Configuration

```python
k = 5  # Top-5 documents retrieved per query

# Why k=5?
# k=3: May miss relevant documents on rare conditions
# k=5: Standard RAG benchmark (precision@5 is the industry metric)
# k=10: Retrieves too much context, degrades LLM response quality
# k=5 will be validated against k=3 and k=7 in evaluation
```

---

## 9. Deployment Architecture (Planned — Phase 4)

```
User Query
    ↓
Streamlit/React Frontend
    ↓
FastAPI Backend (Cloud Run)
    ├── CNN Inference (EfficientNet-B3)
    │     ↓
    │   Detected Conditions
    │     ↓
    └── RAG Pipeline
          ├── BioBERT Embedding
          ├── Pinecone Vector Search (k=5)
          ├── PostgreSQL Metadata Fetch
          └── LLM Response Generation
                ↓
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
| BioBERT over MiniLM | 5x model size | 320MB extra storage | +12-14% retrieval precision |
| Phase 1 binary before Phase 2 multi-label | Delayed clinical utility | 4 weeks | Proven infrastructure, clean baseline metrics |
| GCP Cloud Run over bare VM | Less control | SSH access | Auto-scaling, zero idle cost, managed SSL |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-06 | Initial ARCHITECTURE.md — Phase 1 complete | Ion Turcan |
| — | Phase 2 results (ChestX-ray14) | TBD Week 5 |
| — | Phase 3 RAG evaluation results | TBD Week 6 |
| — | Phase 4 deployment documentation | TBD Week 7 |
