"""Spot-check: inspect actual chunks for TB query + new unseen PE query."""

import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

docs = json.load(open("pipelines/pubmed_cache/documents.json", encoding="utf-8"))
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=80, separators=["\n\n", "\n", ". ", " "]
)
chunks = []
for d in docs:
    chunks.extend(splitter.split_text(d["title"] + "\n\n" + d["abstract"]))

model = SentenceTransformer("all-mpnet-base-v2")
chunk_embs = model.encode(chunks, show_progress_bar=False, batch_size=64)
bm25 = BM25Okapi([c.lower().split() for c in chunks])


def retrieve_best(query, k_init=40, k_final=7, bm25_w=2.0, floor=0.30):
    q_emb = model.encode([query])
    vec_sims = cosine_similarity(q_emb, chunk_embs)[0]
    vec_top = np.argsort(vec_sims)[-k_init:][::-1]
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores)[-k_init:][::-1]
    rrf = defaultdict(float)
    for rank, idx in enumerate(vec_top):
        rrf[idx] += 1.0 / (60 + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] += bm25_w / (60 + rank + 1)
    candidates = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    candidates = [(i, s) for i, s in candidates if vec_sims[i] >= floor]
    top = candidates[:k_final]
    return [(chunks[i], float(vec_sims[i])) for i, _ in top]


queries = [
    {
        "name": "TB QUERY (seen before)",
        "query": "62-year-old male with cavitary lesion in right upper lobe, night sweats, weight loss",
        "topics": [
            "pulmonary tuberculosis with cavitary disease",
            "AFB smear and culture",
            "airborne isolation",
        ],
    },
    {
        "name": "NEW QUERY (never tested)",
        "query": "45-year-old female with sudden onset chest pain and shortness of breath after long flight",
        "topics": [
            "pulmonary embolism diagnosis",
            "D-dimer testing",
            "CT pulmonary angiography",
        ],
    },
]

for q in queries:
    print("=" * 80)
    print(f"TEST: {q['name']}")
    print(f"Query: {q['query']}")
    print(f"Expected: {q['topics']}")
    print()

    results = retrieve_best(q["query"])
    topic_embs = model.encode(q["topics"])

    for rank, (chunk, vsim) in enumerate(results, 1):
        chunk_emb = model.encode([chunk])
        sims = cosine_similarity(chunk_emb, topic_embs)[0]
        best_sim = max(sims)
        best_topic = q["topics"][np.argmax(sims)]
        preview = chunk[:180].replace("\n", " ")

        if best_sim > 0.45:
            status = "RELEVANT"
        elif best_sim < 0.30:
            status = "NOISE"
        else:
            status = "MARGINAL"

        print(f"  #{rank} [{status}] Vec={vsim:.3f} Topic={best_sim:.3f}")
        print(f"    Best match: {best_topic[:50]}")
        print(f"    Text: {preview}...")
        print()

    # Overall score
    all_chunks = [c for c, _ in results]
    ch_embs = model.encode(all_chunks)
    sim_mat = cosine_similarity(ch_embs, topic_embs)
    sp = float(np.mean(sim_mat.max(axis=1)))
    tc_count = sum(1 for s in sim_mat.max(axis=0) if s > 0.45)
    tc_pct = tc_count / len(q["topics"])
    print(f"  SCORE: Sem-P={sp:.3f}, Top-Cov={tc_pct:.1%}")
    print()
