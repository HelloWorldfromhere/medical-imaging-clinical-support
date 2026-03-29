"""
Advanced Retrieval Comparison
Tests 4 retrieval strategies on the medical RAG corpus:
1. Baseline: vector-only with fixed 512 chunks (current system)
2. Natural chunking: paragraph-aware, keep short abstracts whole
3. Hybrid: BM25 keyword search + vector search with RRF fusion
4. Hybrid + Reranking: hybrid retrieval + cross-encoder reranker

Run from project root:
    python test_advanced_retrieval.py
"""

import json
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Try to import BM25 - install if missing
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Installing rank_bm25...")
    import subprocess
    subprocess.check_call(["pip", "install", "rank_bm25", "--break-system-packages"])
    from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading documents...")
docs = json.load(open("pipelines/pubmed_cache/documents.json", encoding="utf-8"))
texts = [d["title"] + "\n\n" + d["abstract"] for d in docs]
print(f"Loaded {len(docs)} documents (avg {int(np.mean([len(t) for t in texts]))} chars)")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load cross-encoder reranker
print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# Test cases (expanded set: 10 cases for better signal)
# ---------------------------------------------------------------------------
test_cases = [
    {
        "query": "65-year-old male with bilateral lobar consolidation, history of COPD, current smoker",
        "topics": [
            "community-acquired pneumonia diagnosis and management",
            "empiric antibiotic therapy selection for pneumonia",
            "COPD exacerbation with respiratory infection",
        ],
    },
    {
        "query": "70-year-old male with large right-sided pleural effusion and mediastinal shift",
        "topics": [
            "thoracentesis indication and pleural fluid analysis",
            "Light criteria for transudative vs exudative effusion",
            "malignant pleural effusion workup",
        ],
    },
    {
        "query": "55-year-old female with widened mediastinum and enlarged cardiac silhouette",
        "topics": [
            "heart failure diagnosis with chest radiograph findings",
            "echocardiogram for ejection fraction assessment",
            "BNP levels in heart failure evaluation",
        ],
    },
    {
        "query": "30-year-old male with spontaneous left pneumothorax, no trauma",
        "topics": [
            "primary spontaneous pneumothorax management",
            "chest tube insertion vs observation criteria",
            "pneumothorax recurrence risk",
        ],
    },
    {
        "query": "80-year-old female with bilateral interstitial opacities and progressive dyspnea",
        "topics": [
            "interstitial lung disease classification",
            "idiopathic pulmonary fibrosis diagnosis",
            "high-resolution CT for fibrosis pattern",
        ],
    },
    {
        "query": "62-year-old male with cavitary lesion in right upper lobe, night sweats, weight loss",
        "topics": [
            "pulmonary tuberculosis with cavitary disease",
            "acid-fast bacilli smear and mycobacterial culture",
            "airborne isolation precautions for TB",
        ],
    },
    {
        "query": "40-year-old immunocompromised male with diffuse ground-glass opacities",
        "topics": [
            "Pneumocystis pneumonia in HIV patients",
            "trimethoprim-sulfamethoxazole treatment",
            "CD4 count and opportunistic infections",
        ],
    },
    {
        "query": "55-year-old female with bilateral pulmonary edema and cardiomegaly, acute onset",
        "topics": [
            "acute decompensated heart failure management",
            "intravenous diuretics for pulmonary edema",
            "noninvasive ventilation with BiPAP",
        ],
    },
    {
        "query": "50-year-old male with bilateral hilar lymphadenopathy, no symptoms",
        "topics": [
            "sarcoidosis as differential diagnosis for hilar adenopathy",
            "serum ACE level and biopsy for granulomas",
            "lymphoma workup for mediastinal lymphadenopathy",
        ],
    },
    {
        "query": "35-year-old female with recurrent bilateral pleural effusions and joint pain",
        "topics": [
            "systemic lupus erythematosus pleuritis",
            "autoimmune workup with ANA and complement",
            "pleural effusion in autoimmune conditions",
        ],
    },
]


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_fixed_512(texts):
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0, separator=" ")
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def chunk_recursive_800(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def chunk_natural_paragraph(texts):
    """
    Keep short documents whole. Only split long ones at paragraph/sentence boundaries.
    This preserves the author's complete thought for most PubMed abstracts.
    """
    chunks = []
    for t in texts:
        if len(t) <= 1000:
            # Short abstract: keep it whole
            chunks.append(t)
        else:
            # Long abstract: split at paragraph boundaries, then sentences
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=90,
                separators=["\n\n", "\n", ". ", " "],
            )
            chunks.extend(splitter.split_text(t))
    return chunks


# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------

def retrieve_vector_only(query, chunk_embs, chunks, model, k=5):
    """Standard vector search — cosine similarity."""
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, chunk_embs)[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in top_idx]


def retrieve_hybrid_rrf(query, chunk_embs, chunks, model, bm25, k=5, rrf_k=60):
    """
    Hybrid: BM25 (keyword) + vector (semantic) fused with Reciprocal Rank Fusion.
    BM25 catches exact medical terms. Vector catches semantic meaning.
    RRF combines rankings without needing to normalize scores.
    """
    # Vector search: top 20
    q_emb = model.encode([query])
    vec_sims = cosine_similarity(q_emb, chunk_embs)[0]
    vec_top = np.argsort(vec_sims)[-20:][::-1]

    # BM25 keyword search: top 20
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[-20:][::-1]

    # Reciprocal Rank Fusion
    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(vec_top):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

    # Sort by fused score, take top k
    fused_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [chunks[idx] for idx, _ in fused_ranked]


def retrieve_hybrid_reranked(query, chunk_embs, chunks, model, bm25, reranker, k=5, rrf_k=60):
    """
    Hybrid retrieval + cross-encoder reranking.
    Step 1: Retrieve top 20 candidates via hybrid (BM25 + vector + RRF)
    Step 2: Rerank those 20 with a cross-encoder that reads query + chunk together
    Step 3: Return top k after reranking
    """
    # Step 1: Get top 20 candidates via hybrid
    q_emb = model.encode([query])
    vec_sims = cosine_similarity(q_emb, chunk_embs)[0]
    vec_top = np.argsort(vec_sims)[-20:][::-1]

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[-20:][::-1]

    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(vec_top):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

    # Top 20 candidates
    candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    candidate_texts = [chunks[idx] for idx, _ in candidates]
    candidate_indices = [idx for idx, _ in candidates]

    # Step 2: Cross-encoder reranking
    pairs = [[query, chunk_text] for chunk_text in candidate_texts]
    rerank_scores = reranker.predict(pairs)

    # Step 3: Sort by reranker score, take top k
    reranked = sorted(zip(candidate_indices, candidate_texts, rerank_scores),
                      key=lambda x: x[2], reverse=True)[:k]
    return [text for _, text, _ in reranked]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(retrieved_chunks, topics, eval_model):
    """Compute semantic precision and topic coverage."""
    if not retrieved_chunks or not topics:
        return 0.0, 0.0

    chunk_embs = eval_model.encode(retrieved_chunks)
    topic_embs = eval_model.encode(topics)
    sim_matrix = cosine_similarity(chunk_embs, topic_embs)

    # Semantic precision: avg of best topic match per chunk
    sem_prec = float(np.mean(sim_matrix.max(axis=1)))

    # Topic coverage: fraction of topics with a chunk above threshold
    topic_max = sim_matrix.max(axis=0)
    coverage = sum(1 for s in topic_max if s > 0.45) / len(topics)

    return round(sem_prec, 4), round(coverage, 4)


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print("ADVANCED RETRIEVAL COMPARISON")
print("=" * 90)

# Prepare chunks for each strategy
strategies = {
    "1. Vector (fixed 512)": ("fixed_512", "vector_only"),
    "2. Vector (recursive 800)": ("recursive_800", "vector_only"),
    "3. Vector (natural paragraph)": ("natural_paragraph", "vector_only"),
    "4. Hybrid BM25+Vec (rec 800)": ("recursive_800", "hybrid"),
    "5. Hybrid+Rerank (rec 800)": ("recursive_800", "hybrid_rerank"),
    "6. Hybrid+Rerank (natural)": ("natural_paragraph", "hybrid_rerank"),
}

chunk_cache = {}
emb_cache = {}
bm25_cache = {}

for name, (chunk_strategy, retrieval_method) in strategies.items():
    # Get or create chunks
    if chunk_strategy not in chunk_cache:
        if chunk_strategy == "fixed_512":
            chunks = chunk_fixed_512(texts)
        elif chunk_strategy == "recursive_800":
            chunks = chunk_recursive_800(texts)
        elif chunk_strategy == "natural_paragraph":
            chunks = chunk_natural_paragraph(texts)
        chunk_cache[chunk_strategy] = chunks
        print(f"\nChunking '{chunk_strategy}': {len(chunks)} chunks, avg {int(np.mean([len(c) for c in chunks]))} chars")

        # Embed
        print(f"  Embedding {len(chunks)} chunks...")
        emb_cache[chunk_strategy] = model.encode(chunks, show_progress_bar=False, batch_size=64)

        # Build BM25 index
        tokenized = [c.lower().split() for c in chunks]
        bm25_cache[chunk_strategy] = BM25Okapi(tokenized)

    chunks = chunk_cache[chunk_strategy]
    chunk_embs = emb_cache[chunk_strategy]
    bm25 = bm25_cache[chunk_strategy]

    # Evaluate across all test cases
    all_sem_prec = []
    all_top_cov = []
    total_time = 0

    for tc in test_cases:
        start = time.perf_counter()

        if retrieval_method == "vector_only":
            retrieved = retrieve_vector_only(tc["query"], chunk_embs, chunks, model, k=5)
        elif retrieval_method == "hybrid":
            retrieved = retrieve_hybrid_rrf(tc["query"], chunk_embs, chunks, model, bm25, k=5)
        elif retrieval_method == "hybrid_rerank":
            retrieved = retrieve_hybrid_reranked(tc["query"], chunk_embs, chunks, model, bm25, reranker, k=5)

        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        sem_prec, top_cov = evaluate_retrieval(retrieved, tc["topics"], model)
        all_sem_prec.append(sem_prec)
        all_top_cov.append(top_cov)

    avg_latency = total_time / len(test_cases)

    print(f"  {name:<35} Sem-P={np.mean(all_sem_prec):.3f}  "
          f"Top-Cov={np.mean(all_top_cov):.3f}  "
          f"Latency={avg_latency:.1f}ms")

print("\n" + "=" * 90)
print("LEGEND:")
print("  Sem-P   = Semantic Precision (higher = chunks more relevant to topics)")
print("  Top-Cov = Topic Coverage (higher = more expected topics found)")
print("  Latency = Time per query including retrieval + reranking")
print("=" * 90)
