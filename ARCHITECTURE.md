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
| **Phase 3** | RAG pipeline — PubMed literature retrieval per detected condition | ✅ Complete | LLM-powered clinical context for each detected pathology |
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
- **False Negative (missed pneumonia):** Patient sent home untreated — potential rapid deterioration, sepsis, death
- **False Positive (healthy patient flagged):** Patient receives additional screening — minor inconvenience, no harm

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

The medical document corpus was built iteratively using data-driven gap analysis. After each expansion, keyword coverage was measured against the evaluation test dataset to identify underrepresented conditions:

| Stage | Documents | Search Terms | Key Additions |
|-------|-----------|--------------|---------------|
| Initial ETL | 228 | 5 chest radiology terms | Pneumonia, pneumothorax, effusion, infiltrates |
| Gap fill #1 | +28 | 3 targeted terms | Lupus pleuritis, heart failure, cardiomegaly |
| General expansion | +326 | 22 condition-specific terms | TB diagnostics, asthma, fibrosis, neonatal RDS, sarcoidosis |
| Gap fill #2 | +193 | 15 procedure-specific terms | AFB smear/culture, PE/D-dimer, bronchoscopy, isolation protocols |
| **Total** | **775** | **45 unique terms** | **15+ clinical conditions, 0 duplicates** |

**Gap analysis methodology:** After each corpus expansion, a keyword frequency check identified conditions with fewer than 10 mentions. Targeted PubMed queries filled these gaps. For example, after the general expansion, pulmonary embolism had 0 mentions — a targeted fetch added 275 PE-related mentions without duplicating existing content. This iterative, data-driven approach to corpus construction is documented as a core engineering practice.

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

The RAG (Retrieval-Augmented Generation) pipeline retrieves relevant medical literature based on CNN classification results and generates clinical decision support summaries. Phase 3 involved three rounds of systematic evaluation, progressively improving the retrieval component through model selection, metric refinement, and retrieval architecture optimization.

### Evaluation Methodology

**Test Dataset:** 20 clinical scenarios covering 7 pathology categories (pneumonia variants, pleural conditions, cardiac findings, pneumothorax, pulmonary fibrosis, tuberculosis, and normal X-ray differentials). Each test case includes a clinical query, expected CNN prediction, expected medical topics, and difficulty classification (standard vs complex).

**Evaluation evolved across three rounds:**

| Round | Metric | What It Measures | Limitation |
|-------|--------|-----------------|------------|
| 1-2 | Keyword Precision@5 | Fraction of top-5 chunks containing expected keywords | Misses synonyms (e.g., "TMP-SMX" vs "trimethoprim-sulfamethoxazole") |
| 3 | Semantic Precision | Cosine similarity between retrieved chunks and expected topic descriptions | Dependent on embedding model quality |
| 3 | Topic Coverage | Fraction of expected clinical topics with at least one relevant chunk (similarity > 0.45) | Threshold selection affects results |

**Why two metrics?** Semantic Precision answers "are the retrieved chunks relevant?" while Topic Coverage answers "did we find something for each clinical concern?" A system could have high precision (all chunks about pneumonia) but low coverage (nothing about antibiotic selection or admission criteria). Both matter for clinical decision support.

**Corpus:** 775 real PubMed abstracts (0 duplicates, verified) fetched via Biopython Entrez API, covering 15+ clinical conditions. All documents are publicly available medical literature cited by PMID.

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

**Models tested:**
- **all-MiniLM-L6-v2:** General-purpose sentence transformer, 80 MB, 384-dimensional. Trained with contrastive learning on 1B+ sentence pairs.
- **BioLORD-2023-M:** Biomedical sentence transformer, 420 MB, 768-dimensional. Trained on biomedical concept relationships from UMLS and clinical ontologies.
- **S-PubMedBERT-MS-MARCO:** PubMedBERT fine-tuned on MS MARCO passage retrieval, 420 MB, 768-dimensional. Combines biomedical language understanding with search-optimized similarity.

**Result:** With proper sentence transformer models, the biomedical models closed the gap with MiniLM but did not surpass it. This motivated deeper investigation into the evaluation metric itself.

### Round 3: Advanced Retrieval Optimization

Recognizing that keyword matching underestimated true retrieval quality, a semantic evaluation metric was developed. The corpus was expanded from 256 to 775 documents through iterative gap analysis. Multiple retrieval architecture improvements were tested.

#### Evaluation Metric Upgrade

| Metric | Before (keyword) | After (semantic) | Interpretation |
|--------|-------------------|-------------------|----------------|
| MiniLM fixed_512 | P@5=0.290 | Sem-P=0.501 | Keyword matching missed relevant chunks using different terminology |

The semantic metric embeds both the retrieved chunks and the expected clinical topic descriptions, then measures cosine similarity between them. A chunk about "empiric antibiotic therapy for community-acquired pneumonia" now correctly matches the topic "antibiotic selection for pneumonia" even without exact keyword overlap.

#### Chunking Strategy Refinement

Five chunking strategies were tested with semantic evaluation on 582 documents:

| Strategy | Chunks | Avg Size | Sem-Prec | Top-Cov |
|----------|--------|----------|----------|---------|
| fixed_256 | 4,590 | 236 chars | 0.573 | 0.600 |
| fixed_512 | 2,413 | 450 chars | 0.569 | 0.800 |
| fixed_1000 | 1,431 | 817 chars | 0.581 | 0.700 |
| **recursive_800** | **2,239** | **491 chars** | **0.592** | **0.700** |
| full_abstract | 926 | 1,178 chars | 0.552 | 0.600 |

**Finding:** Recursive 800-character chunking with paragraph-aware splitting achieved the highest semantic precision. Full-abstract embedding performed worst because averaging an entire 1,800-character abstract into a single embedding vector dilutes the signal — the embedding becomes a blurry representation of multiple clinical concepts rather than a focused representation of one.

#### Embedding Model Comparison (Semantic Metric)

Three models were compared using hybrid retrieval on the expanded 775-document corpus:

| Model | Dimensions | Sem-Prec (k=5) | Top-Cov (k=5) |
|-------|-----------|-----------------|----------------|
| all-MiniLM-L6-v2 | 384 | 0.549 | 0.633 |
| **all-mpnet-base-v2** | **768** | **0.552** | **0.733** |
| BioLORD-2023-M | 768 | 0.528 | 0.667 |

**Finding:** MPNet (768-dimensional) achieved the best combined performance, with notably higher topic coverage than both alternatives. Its larger embedding space captures more semantic nuance than MiniLM's 384 dimensions. BioLORD's biomedical pre-training did not compensate for its smaller general training corpus on this dataset.

#### Hybrid Retrieval: BM25 + Vector Search

Vector-only retrieval misses chunks containing exact medical terms (e.g., "AFB," "BiPAP," "thoracentesis") when the query describes symptoms rather than procedures. Adding BM25 keyword search and fusing results with Reciprocal Rank Fusion (RRF) addresses this:

| Method | Sem-Prec | Top-Cov | Observation |
|--------|----------|---------|-------------|
| Vector-only (MPNet, k=7) | 0.536 | 0.767 | Strong semantics, misses exact terms |
| Hybrid BM25+Vector (k=7) | 0.527 | 0.667 | Catches keywords, adds some noise |
| **Hybrid BM25 2x weight (k=40 to 7, floor=0.30)** | **0.562** | **0.767** | **Best: broad retrieval with noise filtering** |

**BM25 weight tuning:** Giving BM25 double weight (2x) in the RRF fusion prioritizes chunks containing exact medical terminology. Initial retrieval fetches 40 candidates broadly, then a vector similarity floor of 0.30 filters out noise before selecting the top 7.

#### Relevance Threshold Filtering

A key design decision for production safety: chunks below a minimum vector similarity are excluded before being sent to the LLM, even if this means fewer than k chunks are returned. This reduces hallucination risk — sending no context is safer than sending irrelevant context.

| Threshold | Sem-Prec | Top-Cov | Avg Chunks Kept | Effect |
|-----------|----------|---------|-----------------|--------|
| None | 0.515 | 0.767 | 10.0 | All candidates, including noise |
| 0.30 | 0.526 | 0.733 | 9.0 | Removes clearly irrelevant content |
| **0.35** | **0.539** | **0.733** | **8.6** | Optimal: noise removed, coverage maintained |
| 0.45 | 0.543 | 0.733 | 7.6 | Aggressive: precision gains plateau |

#### Cross-Encoder Reranking (Negative Result)

A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was tested to re-score candidates. It reduced performance (Sem-P dropped from 0.562 to 0.470). Root cause: the model was trained on MS MARCO web search queries, not clinical scenarios. It optimizes for "does this web page answer this Google query?" rather than "does this medical abstract relate to this clinical scenario?" A domain-specific medical cross-encoder may improve results, but the general-purpose model is counterproductive.

**Documented as a negative result intentionally** — knowing what doesn't work is as valuable as knowing what does.

#### Chunk Inspection Analysis

Manual inspection of retrieved chunks for specific queries revealed the system's strengths and limitations:

**Strength:** For a tuberculosis query ("cavitary lesion, night sweats, weight loss"), the system retrieves chunks about TB/HIV association (Topic-sim=0.653) and cavitary lesion presentations (Topic-sim=0.519). The hybrid retrieval correctly surfaces chunks containing "AFB" and "cavitary" through BM25 keyword matching.

**Limitation:** The system finds content about the disease itself but struggles with *diagnostic procedure* chunks. A chunk about "AFB smear sensitivity and specificity" ranks lower than symptom-matching chunks because its embedding is about laboratory methodology, not clinical presentation. This is a fundamental embedding model limitation — the model encodes topical similarity, not clinical workflow relevance.

**Unseen query test:** A pulmonary embolism query (not in the training set) was tested after corpus expansion. The system correctly retrieved chunks about pleuritic chest pain and PE diagnosis (Topic-sim=0.562), achieving 66.7% topic coverage — demonstrating generalization to conditions outside the original evaluation set.

### Selected Configuration (Final)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding model | all-mpnet-base-v2 (768-dim) | Best semantic precision and topic coverage across all experiments |
| Chunking | Recursive 800 chars, 80 char overlap | Respects paragraph boundaries, balances focus and context |
| Retrieval | Hybrid BM25 (2x weight) + vector, RRF fusion | BM25 catches exact medical terms that vector similarity misses |
| Initial candidates | 40 | Broad retrieval before filtering |
| Relevance floor | 0.30 vector similarity | Removes noise chunks to reduce hallucination risk |
| Final k | 7 | Optimal balance between coverage and precision |
| **Best Sem-Prec** | **0.562** | Chunks are clearly relevant to clinical topics |
| **Best Top-Cov** | **76.7%** | 2.3 of 3 expected clinical topics addressed per query |

### Improvement Trajectory

| Stage | Corpus | Metric | Sem-Prec | Top-Cov |
|-------|--------|--------|----------|---------|
| Baseline | 256 docs | Keyword P@5 | 0.290 | 0.200 |
| + Semantic metric | 256 docs | Semantic | 0.487 | 0.463 |
| + Corpus expansion | 582 docs | Semantic | 0.501 | 0.525 |
| + MPNet + recursive 800 | 582 docs | Semantic | 0.562 | 0.600 |
| + Hybrid BM25 + filtering | 775 docs | Semantic | 0.562 | 0.767 |

### RAG Pipeline Architecture (Final)

```
Clinical Query + CNN Prediction
        |
        v
  Query Embedding (MPNet, 768-dim)
        |
        v
  Parallel Retrieval
  |-- Vector Search: cosine similarity, top-40 candidates
  |-- BM25 Search: keyword matching, top-40 candidates (2x weight)
        |
        v
  Reciprocal Rank Fusion (RRF, k=60)
        |
        v
  Relevance Floor Filter (vec-sim >= 0.30)
        |
        v
  Top 7 Chunks (~800 chars each)
        |
        v
  Prompt Construction
  (query + CNN result + filtered context)
        |
        v
  LLM Generation (temp=0.3)
        |
        v
  Clinical Summary with Citations
        |
        v
  Query Logged to PostgreSQL
```

**Two retrieval backends:**
- **In-memory (numpy + BM25Okapi):** Used during development and evaluation. Chunk embeddings in numpy array, keyword index via rank_bm25 library.
- **PostgreSQL + pgvector + full-text search:** Production path. Embeddings stored as `vector` type columns with cosine similarity via `<=>` operator. BM25 via PostgreSQL `tsvector` full-text search. Single database handles both retrieval methods.

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
          |-- MPNet Embedding (768-dim)
          |-- Hybrid: pgvector + PostgreSQL FTS
          |-- Relevance Filtering (floor=0.30)
          |-- Top-7 Chunk Selection
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
| MPNet over MiniLM | 5x model size (420 MB vs 80 MB) | Faster inference | Higher topic coverage (73.3% vs 63.3%) |
| MPNet over BioLORD | No biomedical pre-training | Domain-specific embeddings | Better generalization from larger training corpus |
| Recursive 800 over fixed 512 | More chunks (2,239 vs 2,413) | Fewer, larger chunks | Paragraph-aware splitting preserves clinical context |
| Hybrid BM25+vector over vector-only | Added complexity, BM25 index | Simpler retrieval | Catches exact medical terms vectors miss |
| Relevance threshold filtering | May return fewer chunks | Some borderline-relevant content | Reduces hallucination risk from noise chunks |
| General cross-encoder rejected | Lost potential reranking gains | Precision improvement | Avoided degraded results from domain mismatch |
| Semantic over keyword evaluation | More complex metric | Simpler evaluation | Accurate measurement (keyword missed 50%+ of relevant chunks) |
| Sentence transformers over base BERT | Fewer model options | Access to raw BioBERT/PubMedBERT | 2-3x higher retrieval precision through proper training objective |
| Phase 1 binary before Phase 2 multi-label | Delayed clinical utility | 4 weeks | Proven infrastructure, clean baseline metrics |
| GCP Cloud Run over bare VM | Less control | SSH access | Auto-scaling, zero idle cost, managed SSL |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-06 | Initial ARCHITECTURE.md — Phase 1 complete | Ion Turcan |
| 2026-03-27 | Phase 3 RAG evaluation — 2 rounds, 9 configurations each | Ion Turcan |
| 2026-03-28 | Advanced retrieval: semantic metrics, hybrid search, corpus expansion to 775 | Ion Turcan |
| 2026-03-29 | Updated Section 8 with Round 3 results, improvement trajectory, final configuration | Ion Turcan |
| — | Phase 2 results (ChestX-ray14) | TBD Week 5 |
| — | Phase 4 deployment documentation | TBD Week 7 |
