"""
Targeted Retrieval Improvements
Based on chunk inspection findings:
- Chunk #8 (AFB smear TB, most relevant) ranked low due to vector sim mismatch
- Chunk #5 (blastomycosis, irrelevant) was noise at Vec-sim=0.239

Tests:
1. BM25 weight tuning in RRF (give more weight to keyword matches)
2. Vector similarity floor (remove chunks below minimum quality)
3. Retrieve more candidates (k=20 initial, filter down)
4. Combined: weighted RRF + threshold + MPNet

Run: python test_targeted_improvements.py
"""

import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("Loading data and models...")
docs = json.load(open("pipelines/pubmed_cache/documents.json", encoding="utf-8"))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=80, separators=["\n\n", "\n", ". ", " "]
)
chunks = []
for d in docs:
    chunks.extend(splitter.split_text(d["title"] + "\n\n" + d["abstract"]))

bm25 = BM25Okapi([c.lower().split() for c in chunks])
print(f"Corpus: {len(docs)} docs, {len(chunks)} chunks")

# Load both models
model_mini = SentenceTransformer("all-MiniLM-L6-v2")
model_mpnet = SentenceTransformer("all-mpnet-base-v2")

embs_mini = model_mini.encode(chunks, show_progress_bar=False, batch_size=64)
embs_mpnet = model_mpnet.encode(chunks, show_progress_bar=False, batch_size=64)
print("Models and embeddings ready.\n")

# ---------------------------------------------------------------------------
# Test cases
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
        "query": "30-year-old male with spontaneous left pneumothorax, no trauma history",
        "topics": [
            "primary spontaneous pneumothorax management",
            "chest tube insertion vs observation criteria",
            "pneumothorax recurrence risk and prevention",
        ],
    },
    {
        "query": "80-year-old female with bilateral interstitial opacities, progressive dyspnea",
        "topics": [
            "interstitial lung disease classification and diagnosis",
            "idiopathic pulmonary fibrosis treatment options",
            "high-resolution CT for fibrosis pattern assessment",
        ],
    },
    {
        "query": "50-year-old male with bilateral hilar lymphadenopathy, no symptoms",
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
# Retrieval functions
# ---------------------------------------------------------------------------

def retrieve_weighted_hybrid(query, chunk_embs, model, bm25, 
                             initial_k=20, final_k=7,
                             bm25_weight=1.0, vec_weight=1.0,
                             rrf_k=60, min_vec_sim=None):
    """
    Weighted hybrid retrieval with optional vector similarity floor.
    
    bm25_weight / vec_weight control the balance:
      - bm25_weight=2.0 gives keywords 2x importance (catches exact medical terms)
      - vec_weight=2.0 gives semantics 2x importance
    
    min_vec_sim: if set, discard chunks with vector similarity below this threshold
    """
    q_emb = model.encode([query])
    vec_sims = cosine_similarity(q_emb, chunk_embs)[0]
    
    # Vector top candidates
    vec_top = np.argsort(vec_sims)[-initial_k:][::-1]
    
    # BM25 top candidates
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores)[-initial_k:][::-1]
    
    # Weighted RRF fusion
    rrf = defaultdict(float)
    for rank, idx in enumerate(vec_top):
        rrf[idx] += vec_weight / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] += bm25_weight / (rrf_k + rank + 1)
    
    # Sort by fused score
    candidates = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    
    # Apply vector similarity floor if set
    if min_vec_sim is not None:
        candidates = [(idx, score) for idx, score in candidates 
                      if vec_sims[idx] >= min_vec_sim]
    
    # Take top final_k
    top_indices = [idx for idx, _ in candidates[:final_k]]
    top_chunks_text = [chunks[i] for i in top_indices]
    top_vec_sims = [float(vec_sims[i]) for i in top_indices]
    
    return top_chunks_text, top_vec_sims


def evaluate(retrieved_chunks, topics, eval_model):
    if not retrieved_chunks or not topics:
        return 0.0, 0.0, 0
    chunk_embs = eval_model.encode(retrieved_chunks)
    topic_embs = eval_model.encode(topics)
    sim_matrix = cosine_similarity(chunk_embs, topic_embs)
    sem_prec = float(np.mean(sim_matrix.max(axis=1)))
    topic_max = sim_matrix.max(axis=0)
    coverage = sum(1 for s in topic_max if s > 0.45) / len(topics)
    return round(sem_prec, 4), round(coverage, 4), len(retrieved_chunks)


def run_config(model, chunk_embs, bm25_weight, vec_weight, 
               initial_k, final_k, min_vec_sim, label):
    """Run a single configuration across all test cases."""
    all_sp, all_tc, all_kept = [], [], []
    for tc in test_cases:
        ret_chunks, ret_sims = retrieve_weighted_hybrid(
            tc["query"], chunk_embs, model, bm25,
            initial_k=initial_k, final_k=final_k,
            bm25_weight=bm25_weight, vec_weight=vec_weight,
            min_vec_sim=min_vec_sim,
        )
        sp, cov, kept = evaluate(ret_chunks, tc["topics"], model)
        all_sp.append(sp)
        all_tc.append(cov)
        all_kept.append(kept)
    
    avg_sp = np.mean(all_sp)
    avg_tc = np.mean(all_tc)
    avg_kept = np.mean(all_kept)
    print(f"  {label:<50} Sem-P={avg_sp:.3f}  Top-Cov={avg_tc:.3f}  Kept={avg_kept:.1f}")
    return avg_sp, avg_tc


# ---------------------------------------------------------------------------
# Experiment 1: BM25 weight tuning (MPNet)
# ---------------------------------------------------------------------------
print("=" * 85)
print("EXPERIMENT 1: BM25 Weight in Hybrid Fusion (MPNet)")
print("  Higher BM25 weight = more emphasis on exact medical keywords")
print("=" * 85)

for bm25_w in [0.5, 1.0, 1.5, 2.0, 3.0]:
    label = f"BM25={bm25_w:.1f} / Vec=1.0, k=7"
    run_config(model_mpnet, embs_mpnet, bm25_w, 1.0, 20, 7, None, label)

# ---------------------------------------------------------------------------
# Experiment 2: Retrieve broad, filter tight (MPNet)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 85}")
print("EXPERIMENT 2: Retrieve Broad (k=15), Filter by Vec-Sim Threshold (MPNet)")
print("  Cast wide net then remove noise chunks")
print("=" * 85)

for min_sim in [None, 0.25, 0.30, 0.35, 0.40]:
    thresh_label = f"{min_sim:.2f}" if min_sim else "None"
    label = f"Initial=15, Threshold={thresh_label}"
    run_config(model_mpnet, embs_mpnet, 1.0, 1.0, 20, 15, min_sim, label)

# ---------------------------------------------------------------------------
# Experiment 3: Combined - best BM25 weight + threshold (MPNet)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 85}")
print("EXPERIMENT 3: Combined Optimizations (MPNet)")
print("  Best BM25 weight + threshold + different final k")
print("=" * 85)

configs = [
    # (bm25_w, vec_w, initial_k, final_k, min_sim, label)
    (1.0, 1.0, 20, 5, None,  "Baseline: equal weights, k=5, no filter"),
    (1.5, 1.0, 20, 7, None,  "BM25 boosted, k=7, no filter"),
    (1.5, 1.0, 20, 7, 0.30,  "BM25 boosted, k=7, floor=0.30"),
    (2.0, 1.0, 30, 10, 0.30, "BM25 2x, broad k=30->10, floor=0.30"),
    (2.0, 1.0, 30, 10, 0.35, "BM25 2x, broad k=30->10, floor=0.35"),
    (1.5, 1.0, 30, 7, 0.30,  "BM25 1.5x, broad k=30->7, floor=0.30"),
    (2.0, 1.0, 40, 7, 0.30,  "BM25 2x, very broad k=40->7, floor=0.30"),
]

best_sp = 0
best_tc = 0
best_label = ""

for bm25_w, vec_w, ik, fk, ms, label in configs:
    sp, tc = run_config(model_mpnet, embs_mpnet, bm25_w, vec_w, ik, fk, ms, label)
    # Best = highest combined score
    combined = sp * 0.5 + tc * 0.5
    if combined > best_sp * 0.5 + best_tc * 0.5:
        best_sp, best_tc, best_label = sp, tc, label

# ---------------------------------------------------------------------------
# Experiment 4: Same best config with MiniLM for comparison
# ---------------------------------------------------------------------------
print(f"\n{'=' * 85}")
print("EXPERIMENT 4: Best Config on MiniLM vs MPNet")
print("=" * 85)

run_config(model_mini, embs_mini, 2.0, 1.0, 30, 10, 0.30, "MiniLM: BM25 2x, k=30->10, floor=0.30")
run_config(model_mpnet, embs_mpnet, 2.0, 1.0, 30, 10, 0.30, "MPNet:  BM25 2x, k=30->10, floor=0.30")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 85}")
print(f"BEST CONFIGURATION: {best_label}")
print(f"  Semantic Precision: {best_sp:.3f}")
print(f"  Topic Coverage:     {best_tc:.3f}")
print(f"{'=' * 85}")
