"""
Final Retrieval Optimization
Tests the best ideas from all previous experiments:
1. Relevance threshold filtering (discard low-quality chunks)
2. MPNet (768-dim) vs BioLORD (biomedical sentence transformer)
3. Hybrid retrieval + filtering
4. Different threshold values to find the sweet spot

Run: python test_final_optimization.py
"""

import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading documents...")
docs = json.load(open("pipelines/pubmed_cache/documents.json", encoding="utf-8"))
print(f"Loaded {len(docs)} documents")

# ---------------------------------------------------------------------------
# Test cases (10 diverse clinical scenarios)
# ---------------------------------------------------------------------------
test_cases = [
    {
        "query": "65-year-old male with bilateral lobar consolidation, history of COPD",
        "topics": [
            "community-acquired pneumonia diagnosis and management",
            "empiric antibiotic therapy selection for pneumonia",
            "COPD exacerbation with respiratory infection",
        ],
    },
    {
        "query": "70-year-old with large right-sided pleural effusion and mediastinal shift",
        "topics": [
            "thoracentesis indication and pleural fluid analysis",
            "Light criteria for transudative vs exudative effusion",
            "malignant pleural effusion workup",
        ],
    },
    {
        "query": "55-year-old female with enlarged cardiac silhouette, acute onset dyspnea",
        "topics": [
            "heart failure diagnosis with chest radiograph",
            "echocardiogram for ejection fraction assessment",
            "BNP levels in heart failure evaluation",
        ],
    },
    {
        "query": "62-year-old male with cavitary lesion in right upper lobe, night sweats",
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
        "query": "30-year-old male with spontaneous left pneumothorax, no trauma history",
        "topics": [
            "primary spontaneous pneumothorax management",
            "chest tube insertion vs observation criteria",
            "pneumothorax recurrence risk and prevention",
        ],
    },
    {
        "query": "80-year-old female with bilateral interstitial opacities, progressive dyspnea over months",
        "topics": [
            "interstitial lung disease classification and diagnosis",
            "idiopathic pulmonary fibrosis treatment options",
            "high-resolution CT for fibrosis pattern assessment",
        ],
    },
    {
        "query": "50-year-old male with bilateral hilar lymphadenopathy, no respiratory symptoms",
        "topics": [
            "sarcoidosis differential diagnosis for hilar adenopathy",
            "serum ACE level and tissue biopsy for granulomas",
            "lymphoma workup for mediastinal lymphadenopathy",
        ],
    },
    {
        "query": "35-year-old female with recurrent bilateral pleural effusions and joint pain",
        "topics": [
            "systemic lupus erythematosus pleuritis",
            "autoimmune workup with ANA and complement levels",
            "pleural effusion in autoimmune conditions",
        ],
    },
    {
        "query": "55-year-old female with bilateral pulmonary edema and cardiomegaly",
        "topics": [
            "acute decompensated heart failure management",
            "intravenous diuretics for pulmonary edema",
            "noninvasive ventilation with BiPAP or CPAP",
        ],
    },
]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=80, separators=["\n\n", "\n", ". ", " "]
)
chunks = []
for d in docs:
    full_text = d["title"] + "\n\n" + d["abstract"]
    chunks.extend(splitter.split_text(full_text))
print(f"Chunks: {len(chunks)}, avg {int(np.mean([len(c) for c in chunks]))} chars")

# BM25
bm25 = BM25Okapi([c.lower().split() for c in chunks])


# ---------------------------------------------------------------------------
# Evaluation with relevance threshold
# ---------------------------------------------------------------------------

def evaluate_with_threshold(retrieved_chunks, retrieved_sims, topics, eval_model, threshold=None):
    """
    Evaluate retrieval, optionally filtering chunks below similarity threshold.
    If threshold is set, only chunks with similarity >= threshold are kept.
    If no chunks pass the filter, metrics are 0 (but this is SAFER than hallucination).
    """
    if threshold is not None:
        # Filter: only keep chunks above the quality threshold
        filtered = [(c, s) for c, s in zip(retrieved_chunks, retrieved_sims)
                     if s >= threshold]
        if not filtered:
            return 0.0, 0.0, 0  # No chunks passed filter
        filtered_chunks = [c for c, s in filtered]
        n_kept = len(filtered_chunks)
    else:
        filtered_chunks = retrieved_chunks
        n_kept = len(filtered_chunks)

    chunk_embs = eval_model.encode(filtered_chunks)
    topic_embs = eval_model.encode(topics)
    sim_matrix = cosine_similarity(chunk_embs, topic_embs)

    sem_prec = float(np.mean(sim_matrix.max(axis=1)))
    topic_max = sim_matrix.max(axis=0)
    coverage = sum(1 for s in topic_max if s > 0.45) / len(topics)

    return round(sem_prec, 4), round(coverage, 4), n_kept


def retrieve_hybrid(query, chunk_embs, chunks, model, bm25, k=10):
    """Retrieve top-k via hybrid BM25+vector with RRF. Returns chunks + similarity scores."""
    q_emb = model.encode([query])
    vec_sims = cosine_similarity(q_emb, chunk_embs)[0]

    vec_top = np.argsort(vec_sims)[-20:][::-1]
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores)[-20:][::-1]

    rrf = defaultdict(float)
    for rank, idx in enumerate(vec_top):
        rrf[idx] += 1.0 / (60 + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] += 1.0 / (60 + rank + 1)

    top_indices = [idx for idx, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]]
    top_chunks = [chunks[i] for i in top_indices]
    top_sims = [float(vec_sims[i]) for i in top_indices]  # vector similarity for filtering

    return top_chunks, top_sims


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------
models_to_test = {
    "MiniLM (384-dim)": "all-MiniLM-L6-v2",
    "MPNet (768-dim)": "all-mpnet-base-v2",
    "BioLORD (768-dim, biomedical)": "FremyCompany/BioLORD-2023-M",
}

for model_name, model_id in models_to_test.items():
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    model = SentenceTransformer(model_id)
    chunk_embs = model.encode(chunks, show_progress_bar=False, batch_size=64)

    # Test 1: Different k without threshold (baseline)
    print(f"\n  --- Hybrid retrieval, no threshold ---")
    print(f"  {'k':>4}  {'Sem-Prec':>10}  {'Top-Cov':>10}")
    print(f"  {'-'*30}")
    for k in [5, 7, 10]:
        all_sp, all_tc = [], []
        for tc in test_cases:
            ret_chunks, ret_sims = retrieve_hybrid(tc["query"], chunk_embs, chunks, model, bm25, k=k)
            sp, cov, _ = evaluate_with_threshold(ret_chunks, ret_sims, tc["topics"], model)
            all_sp.append(sp)
            all_tc.append(cov)
        print(f"  {k:>4}  {np.mean(all_sp):>10.3f}  {np.mean(all_tc):>10.3f}")

    # Test 2: k=10 with different thresholds (retrieve broad, filter tight)
    print(f"\n  --- Hybrid k=10, with relevance threshold ---")
    print(f"  {'Threshold':>10}  {'Sem-Prec':>10}  {'Top-Cov':>10}  {'Avg Kept':>10}")
    print(f"  {'-'*45}")
    for threshold in [None, 0.20, 0.30, 0.35, 0.40, 0.45]:
        all_sp, all_tc, all_kept = [], [], []
        for tc in test_cases:
            ret_chunks, ret_sims = retrieve_hybrid(tc["query"], chunk_embs, chunks, model, bm25, k=10)
            sp, cov, kept = evaluate_with_threshold(
                ret_chunks, ret_sims, tc["topics"], model, threshold=threshold
            )
            all_sp.append(sp)
            all_tc.append(cov)
            all_kept.append(kept)
        thresh_label = f"{threshold:.2f}" if threshold else "None"
        print(f"  {thresh_label:>10}  {np.mean(all_sp):>10.3f}  {np.mean(all_tc):>10.3f}  {np.mean(all_kept):>10.1f}")

print(f"\n{'='*80}")
print("INTERPRETATION:")
print("  - Higher Sem-Prec with threshold = filtering removes noise, reduces hallucination risk")
print("  - Lower Avg Kept = fewer chunks sent to LLM, more focused context")
print("  - If Top-Cov drops sharply, threshold is too aggressive")
print("  - Sweet spot: threshold where Sem-Prec increases without killing Top-Cov")
print(f"{'='*80}")
