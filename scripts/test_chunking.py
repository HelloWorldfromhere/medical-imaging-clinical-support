"""Quick chunking strategy comparison - run from project root."""

import json

import numpy as np
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load docs
docs = json.load(open("pipelines/pubmed_cache/documents.json", encoding="utf-8"))
texts = [d["title"] + "\n\n" + d["abstract"] for d in docs]
print(f"Loaded {len(docs)} documents")

sizes = [len(t) for t in texts]
print(f"Document sizes: min={min(sizes)}, avg={int(np.mean(sizes))}, max={max(sizes)}")

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5 representative test cases
test_queries = [
    "community-acquired pneumonia diagnosis and empiric antibiotic therapy",
    "pleural effusion thoracentesis Light criteria",
    "acute decompensated heart failure BNP diuretics pulmonary edema",
    "pulmonary tuberculosis cavitary lesion AFB culture isolation",
    "primary spontaneous pneumothorax chest tube observation",
]
test_topics = [
    ["empiric antibiotic therapy selection for pneumonia", "criteria for hospital admission"],
    ["thoracentesis indication and pleural fluid analysis", "Light criteria for effusion"],
    ["heart failure diagnosis with chest radiograph", "diuretic therapy for fluid overload"],
    ["acid-fast bacilli smear and mycobacterial culture", "airborne isolation precautions"],
    ["pneumothorax size estimation on radiograph", "chest tube vs observation criteria"],
]

strategies = {
    "fixed_512": CharacterTextSplitter(chunk_size=512, chunk_overlap=0, separator=" "),
    "recursive_600": RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
    ),
    "recursive_800": RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, separators=["\n\n", "\n", ". ", " "]
    ),
    "recursive_1000": RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
    ),
    "recursive_1200": RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "]
    ),
}

header = f"{'Strategy':<20} {'Chunks':>7} {'Avg Size':>9} {'Sem-Prec':>10} {'Top-Cov':>9}"
print(f"\n{header}")
print("-" * 60)

for name, splitter in strategies.items():
    all_chunks = []
    for t in texts:
        all_chunks.extend(splitter.split_text(t))

    chunk_embs = model.encode(all_chunks, show_progress_bar=False, batch_size=64)

    sem_precs = []
    top_covs = []

    for query, topics in zip(test_queries, test_topics):
        q_emb = model.encode([query])
        sims = cosine_similarity(q_emb, chunk_embs)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]
        top5_texts = [all_chunks[i] for i in top5_idx]

        topic_embs = model.encode(topics)
        chunk5_embs = model.encode(top5_texts)
        sim_matrix = cosine_similarity(chunk5_embs, topic_embs)

        sem_precs.append(float(np.mean(sim_matrix.max(axis=1))))
        covered = sum(1 for s in sim_matrix.max(axis=0) if s > 0.45)
        top_covs.append(covered / len(topics))

    avg_size = int(np.mean([len(c) for c in all_chunks]))
    print(f"{name:<20} {len(all_chunks):>7} {avg_size:>9} {np.mean(sem_precs):>10.3f} {np.mean(top_covs):>9.3f}")

print()
