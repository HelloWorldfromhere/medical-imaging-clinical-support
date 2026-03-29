"""
Test all suggested improvements:
1. Different k values (3, 5, 7, 10)
2. Contextual chunk prefixes (prepend document title)
3. Larger embedding model (all-mpnet-base-v2, 768-dim)
4. Hybrid retrieval with best settings

Run: python test_improvements.py
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
        "query": "55-year-old female with enlarged cardiac silhouette, acute onset",
        "topics": [
            "heart failure diagnosis with chest radiograph",
            "echocardiogram for ejection fraction assessment",
            "BNP levels in heart failure evaluation",
        ],
    },
    {
        "query": "62-year-old male with cavitary lesion, night sweats, weight loss",
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
        "query": "30-year-old male with spontaneous left pneumothorax, no trauma",
        "topics": [
            "primary spontaneous pneumothorax management",
            "chest tube insertion vs observation criteria",
            "pneumothorax recurrence risk",
        ],
    },
    {
        "query": "80-year-old female with bilateral interstitial opacities, progressive dyspnea",
        "topics": [
            "interstitial lung disease classification",
            "idiopathic pulmonary fibrosis diagnosis",
            "high-resolution CT for fibrosis pattern",
        ],
    },
    {
        "query": "50-year-old male with bilateral hilar lymphadenopathy, no symptoms",
        "topics": [
            "sarcoidosis differential diagnosis",
            "serum ACE level and granuloma biopsy",
            "lymphoma workup for mediastinal adenopathy",
        ],
    },
    {
        "query": "35-year-old female with recurrent pleural effusions and joint pain",
        "topics": [
            "systemic lupus erythematosus pleuritis",
            "autoimmune workup with ANA and complement",
            "pleural effusion in autoimmune conditions",
        ],
    },
    {
        "query": "55-year-old female with bilateral pulmonary edema and cardiomegaly",
        "topics": [
            "acute decompensated heart failure management",
            "intravenous diuretics for pulmonary edema",
            "noninvasive ventilation with BiPAP",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def evaluate(retrieved_chunks, topics, eval_model):
    if not retrieved_chunks or not topics:
        return 0.0, 0.0
    chunk_embs = eval_model.encode(retrieved_chunks)
    topic_embs = eval_model.encode(topics)
    sim_matrix = cosine_similarity(chunk_embs, topic_embs)
    sem_prec = float(np.mean(sim_matrix.max(axis=1)))
    topic_max = sim_matrix.max(axis=0)
    coverage = sum(1 for s in topic_max if s > 0.45) / len(topics)
    return round(sem_prec, 4), round(coverage, 4)


def run_eval(chunks, chunk_embs, eval_model, k=5, bm25=None, hybrid=False):
    """Run evaluation across all test cases."""
    all_sp = []
    all_tc = []
    for tc in test_cases:
        query = tc["query"]
        q_emb = eval_model.encode([query])
        vec_sims = cosine_similarity(q_emb, chunk_embs)[0]

        if hybrid and bm25:
            # Hybrid: BM25 + vector with RRF
            vec_top = np.argsort(vec_sims)[-20:][::-1]
            bm25_scores = bm25.get_scores(query.lower().split())
            bm25_top = np.argsort(bm25_scores)[-20:][::-1]
            rrf = defaultdict(float)
            for rank, idx in enumerate(vec_top):
                rrf[idx] += 1.0 / (60 + rank + 1)
            for rank, idx in enumerate(bm25_top):
                rrf[idx] += 1.0 / (60 + rank + 1)
            top_idx = [idx for idx, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]]
        else:
            top_idx = np.argsort(vec_sims)[-k:][::-1]

        retrieved = [chunks[i] for i in top_idx]
        sp, tc_score = evaluate(retrieved, tc["topics"], eval_model)
        all_sp.append(sp)
        all_tc.append(tc_score)
    return round(np.mean(all_sp), 3), round(np.mean(all_tc), 3)


# ---------------------------------------------------------------------------
# Prepare chunks
# ---------------------------------------------------------------------------

# Standard chunks (no prefix)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=80, separators=["\n\n", "\n", ". ", " "]
)
texts = [d["title"] + "\n\n" + d["abstract"] for d in docs]
chunks_plain = []
for t in texts:
    chunks_plain.extend(splitter.split_text(t))

# Contextual chunks (prepend title as prefix)
chunks_contextual = []
for d in docs:
    full_text = d["title"] + "\n\n" + d["abstract"]
    splits = splitter.split_text(full_text)
    for s in splits:
        # Prepend title so each chunk knows what paper it's from
        prefix = f"[{d['title'][:80]}] "
        chunks_contextual.append(prefix + s)

print(f"Plain chunks: {len(chunks_plain)}")
print(f"Contextual chunks: {len(chunks_contextual)}")

# BM25 indexes
bm25_plain = BM25Okapi([c.lower().split() for c in chunks_plain])
bm25_contextual = BM25Okapi([c.lower().split() for c in chunks_contextual])


# ---------------------------------------------------------------------------
# Test 1: Different k values with MiniLM
# ---------------------------------------------------------------------------
print("\n--- TEST 1: Effect of k (number of retrieved chunks) ---")
print("Model: all-MiniLM-L6-v2, Strategy: recursive 800, Method: vector-only\n")

model_mini = SentenceTransformer("all-MiniLM-L6-v2")
embs_plain_mini = model_mini.encode(chunks_plain, show_progress_bar=False, batch_size=64)

header = f"{'k':>4}  {'Sem-Prec':>10}  {'Top-Cov':>10}  {'Interpretation'}"
print(header)
print("-" * 65)
for k in [3, 5, 7, 10, 15]:
    sp, tc = run_eval(chunks_plain, embs_plain_mini, model_mini, k=k)
    if k == 3:
        note = "Focused but may miss topics"
    elif k == 5:
        note = "Current default"
    elif k == 7:
        note = "Wider net"
    elif k == 10:
        note = "Broad coverage, more noise"
    else:
        note = "Very broad, risk of noise"
    print(f"{k:>4}  {sp:>10.3f}  {tc:>10.3f}  {note}")


# ---------------------------------------------------------------------------
# Test 2: Contextual prefix vs plain
# ---------------------------------------------------------------------------
print("\n--- TEST 2: Contextual chunk prefix (prepend title) ---")
print("Model: all-MiniLM-L6-v2, k=7, Strategy: recursive 800\n")

embs_ctx_mini = model_mini.encode(chunks_contextual, show_progress_bar=False, batch_size=64)

sp_plain, tc_plain = run_eval(chunks_plain, embs_plain_mini, model_mini, k=7)
sp_ctx, tc_ctx = run_eval(chunks_contextual, embs_ctx_mini, model_mini, k=7)

print(f"{'Method':<25} {'Sem-Prec':>10} {'Top-Cov':>10}")
print("-" * 50)
print(f"{'Plain chunks':<25} {sp_plain:>10.3f} {tc_plain:>10.3f}")
print(f"{'With title prefix':<25} {sp_ctx:>10.3f} {tc_ctx:>10.3f}")


# ---------------------------------------------------------------------------
# Test 3: Larger embedding model
# ---------------------------------------------------------------------------
print("\n--- TEST 3: Larger embedding model ---")
print("Strategy: recursive 800 + contextual prefix, k=7\n")

print("Loading all-mpnet-base-v2 (768-dim)...")
model_mpnet = SentenceTransformer("all-mpnet-base-v2")
embs_ctx_mpnet = model_mpnet.encode(chunks_contextual, show_progress_bar=False, batch_size=64)

sp_mpnet, tc_mpnet = run_eval(chunks_contextual, embs_ctx_mpnet, model_mpnet, k=7)

print(f"{'Model':<30} {'Dim':>5} {'Sem-Prec':>10} {'Top-Cov':>10}")
print("-" * 60)
print(f"{'MiniLM (current)':<30} {'384':>5} {sp_ctx:>10.3f} {tc_ctx:>10.3f}")
print(f"{'MPNet (larger)':<30} {'768':>5} {sp_mpnet:>10.3f} {tc_mpnet:>10.3f}")


# ---------------------------------------------------------------------------
# Test 4: Best combination — hybrid + best settings
# ---------------------------------------------------------------------------
print("\n--- TEST 4: Best combination ---")
print("Hybrid BM25+Vector, contextual prefix, k=7\n")

# Use whichever model performed better
best_model = model_mpnet if sp_mpnet > sp_ctx else model_mini
best_embs = embs_ctx_mpnet if sp_mpnet > sp_ctx else embs_ctx_mini
best_name = "MPNet" if sp_mpnet > sp_ctx else "MiniLM"

sp_hybrid, tc_hybrid = run_eval(
    chunks_contextual, best_embs, best_model, k=7,
    bm25=bm25_contextual, hybrid=True
)

print(f"{'Configuration':<45} {'Sem-Prec':>10} {'Top-Cov':>10}")
print("-" * 70)
print(f"{'Baseline (MiniLM, plain, vec, k=5)':<45} {sp_plain:>10.3f} {tc_plain:>10.3f}")
print(f"{'Best ({best_name}, ctx prefix, hybrid, k=7)':<45} {sp_hybrid:>10.3f} {tc_hybrid:>10.3f}")

improvement_sp = ((sp_hybrid - sp_plain) / sp_plain) * 100 if sp_plain > 0 else 0
improvement_tc = ((tc_hybrid - tc_plain) / tc_plain) * 100 if tc_plain > 0 else 0
print(f"\nImprovement: Sem-Prec {improvement_sp:+.1f}%, Top-Cov {improvement_tc:+.1f}%")

print("\nDone.")
