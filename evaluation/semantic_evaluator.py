"""
Semantic RAG Evaluator — Embedding-based retrieval quality measurement
Medical Imaging RAG Clinical Decision Support

Improves on keyword matching by using cosine similarity between
retrieved chunks and expected topic descriptions. A chunk about
"trimethoprim-sulfamethoxazole therapy" now correctly matches
the expected topic "TMP-SMX treatment" because their embeddings
are close in vector space.

Usage:
    python -m evaluation.semantic_evaluator                  # Full evaluation
    python -m evaluation.semantic_evaluator --model minilm   # Single model
    python -m evaluation.semantic_evaluator --compare        # Side-by-side with keyword metric
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag.embedding_pipeline import (
    CHUNKING_STRATEGIES,
    EMBEDDING_MODELS,
    EmbeddingPipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enhanced test cases with topic descriptions for semantic matching
# ---------------------------------------------------------------------------

SEMANTIC_TEST_CASES = [
    {
        "id": 1,
        "query": "65-year-old male with bilateral lobar consolidation on chest X-ray, history of COPD and current smoker",
        "expected_topics": [
            "community-acquired pneumonia diagnosis and management",
            "empiric antibiotic therapy selection for pneumonia",
            "COPD exacerbation with respiratory infection",
            "criteria for hospital admission in pneumonia",
        ],
        "difficulty": "standard",
    },
    {
        "id": 2,
        "query": "45-year-old female with diffuse bilateral infiltrates, fever, and hypoxia, recently hospitalized",
        "expected_topics": [
            "hospital-acquired or nosocomial pneumonia",
            "MRSA coverage and broad-spectrum antibiotics",
            "ventilator-associated pneumonia risk factors",
            "respiratory culture and sensitivity testing",
        ],
        "difficulty": "complex",
    },
    {
        "id": 3,
        "query": "70-year-old male with large right-sided pleural effusion and mediastinal shift",
        "expected_topics": [
            "thoracentesis indication and pleural fluid analysis",
            "Light criteria for transudative vs exudative effusion",
            "malignant pleural effusion workup",
            "chest tube drainage for large effusions",
        ],
        "difficulty": "standard",
    },
    {
        "id": 4,
        "query": "55-year-old female with widened mediastinum and enlarged cardiac silhouette",
        "expected_topics": [
            "heart failure diagnosis with chest radiograph findings",
            "echocardiogram for ejection fraction assessment",
            "BNP and NT-proBNP in heart failure evaluation",
            "diuretic therapy for fluid overload",
        ],
        "difficulty": "standard",
    },
    {
        "id": 5,
        "query": "30-year-old male with spontaneous left pneumothorax on chest X-ray, no trauma history",
        "expected_topics": [
            "primary spontaneous pneumothorax management",
            "chest tube insertion vs observation criteria",
            "pneumothorax size estimation on radiograph",
            "recurrence risk and prevention strategies",
        ],
        "difficulty": "standard",
    },
    {
        "id": 6,
        "query": "60-year-old female with right upper lobe mass and hilar lymphadenopathy, 40 pack-year smoking history",
        "expected_topics": [
            "lung cancer staging with CT and PET imaging",
            "tissue biopsy for histological diagnosis",
            "non-small cell lung cancer classification",
            "mediastinal lymph node evaluation",
        ],
        "difficulty": "complex",
    },
    {
        "id": 7,
        "query": "80-year-old female with bilateral interstitial opacities and progressive dyspnea over 6 months",
        "expected_topics": [
            "interstitial lung disease classification",
            "idiopathic pulmonary fibrosis diagnosis",
            "high-resolution CT for fibrosis pattern assessment",
            "pulmonary function testing with restrictive pattern",
        ],
        "difficulty": "complex",
    },
    {
        "id": 8,
        "query": "25-year-old asthmatic female with hyperinflated lungs and flattened diaphragms during acute exacerbation",
        "expected_topics": [
            "asthma exacerbation management and bronchodilator therapy",
            "systemic corticosteroids for acute asthma",
            "peak flow and oxygen saturation monitoring",
            "chest radiograph findings in acute asthma",
        ],
        "difficulty": "standard",
    },
    {
        "id": 9,
        "query": "50-year-old male with bilateral hilar lymphadenopathy and no respiratory symptoms",
        "expected_topics": [
            "sarcoidosis as differential diagnosis",
            "lymphoma workup for mediastinal adenopathy",
            "serum ACE level and calcium testing",
            "tissue biopsy for noncaseating granulomas",
        ],
        "difficulty": "complex",
    },
    {
        "id": 10,
        "query": "72-year-old male with right lower lobe atelectasis post-abdominal surgery",
        "expected_topics": [
            "post-operative atelectasis prevention and management",
            "incentive spirometry and early mobilization",
            "mucus plugging and bronchoscopy indication",
            "chest physiotherapy for lung re-expansion",
        ],
        "difficulty": "standard",
    },
    {
        "id": 11,
        "query": "40-year-old immunocompromised male with diffuse ground-glass opacities bilaterally",
        "expected_topics": [
            "Pneumocystis pneumonia in HIV patients",
            "trimethoprim-sulfamethoxazole treatment for PCP",
            "CD4 count correlation with opportunistic infections",
            "bronchoalveolar lavage for diagnosis",
        ],
        "difficulty": "complex",
    },
    {
        "id": 12,
        "query": "Neonate with bilateral diffuse granular opacities and air bronchograms, born at 28 weeks",
        "expected_topics": [
            "neonatal respiratory distress syndrome from surfactant deficiency",
            "exogenous surfactant administration therapy",
            "continuous positive airway pressure for premature infants",
            "antenatal corticosteroids for lung maturity",
        ],
        "difficulty": "complex",
    },
    {
        "id": 13,
        "query": "58-year-old male with unilateral left pleural effusion and rib fractures after motor vehicle accident",
        "expected_topics": [
            "traumatic hemothorax assessment and management",
            "chest tube thoracostomy for hemothorax",
            "CT thorax for injury assessment after trauma",
            "surgical exploration criteria for massive hemothorax",
        ],
        "difficulty": "standard",
    },
    {
        "id": 14,
        "query": "35-year-old female with recurrent small bilateral pleural effusions and joint pain",
        "expected_topics": [
            "systemic lupus erythematosus pleuritis",
            "autoimmune workup with ANA and complement levels",
            "rheumatology referral for connective tissue disease",
            "pleural effusion in autoimmune conditions",
        ],
        "difficulty": "complex",
    },
    {
        "id": 15,
        "query": "68-year-old male with right middle lobe consolidation not responding to 2 weeks of antibiotics",
        "expected_topics": [
            "non-resolving pneumonia differential diagnosis",
            "post-obstructive pneumonia from endobronchial lesion",
            "bronchoscopy for persistent consolidation",
            "malignancy workup with CT chest",
        ],
        "difficulty": "complex",
    },
    {
        "id": 16,
        "query": "22-year-old female with normal chest X-ray but persistent cough and wheezing for 3 months",
        "expected_topics": [
            "spirometry and methacholine challenge for asthma diagnosis",
            "gastroesophageal reflux as cause of chronic cough",
            "post-nasal drip syndrome evaluation",
            "normal radiograph differential for chronic cough",
        ],
        "difficulty": "standard",
    },
    {
        "id": 17,
        "query": "75-year-old male with tension pneumothorax, tracheal deviation, and hemodynamic instability",
        "expected_topics": [
            "tension pneumothorax emergency management",
            "needle decompression as immediate intervention",
            "chest tube insertion after decompression",
            "hemodynamic support during resuscitation",
        ],
        "difficulty": "complex",
    },
    {
        "id": 18,
        "query": "48-year-old obese female with elevated right hemidiaphragm and basilar atelectasis",
        "expected_topics": [
            "obesity-related atelectasis and respiratory mechanics",
            "diaphragm dysfunction evaluation",
            "positional management and CPAP therapy",
            "weight management for pulmonary improvement",
        ],
        "difficulty": "standard",
    },
    {
        "id": 19,
        "query": "62-year-old male with cavitary lesion in right upper lobe, night sweats, and weight loss",
        "expected_topics": [
            "pulmonary tuberculosis with cavitary disease",
            "acid-fast bacilli smear and mycobacterial culture",
            "airborne isolation precautions for suspected TB",
            "anti-tuberculosis drug regimen",
        ],
        "difficulty": "complex",
    },
    {
        "id": 20,
        "query": "55-year-old female with bilateral pulmonary edema pattern and cardiomegaly, acute onset",
        "expected_topics": [
            "acute decompensated heart failure management",
            "intravenous diuretics for pulmonary edema",
            "noninvasive ventilation with BiPAP or CPAP",
            "troponin measurement to rule out myocardial infarction",
        ],
        "difficulty": "standard",
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SemanticScore:
    test_case_id: int
    query: str
    model_key: str
    strategy_key: str
    semantic_precision: float    # avg similarity of top-5 to best-matching topic
    topic_coverage: float        # fraction of topics with a relevant chunk (sim > threshold)
    best_chunk_similarity: float # highest similarity between any chunk and any topic
    keyword_precision: float     # original keyword metric for comparison
    latency_ms: float


@dataclass
class SemanticConfigScore:
    model_key: str
    strategy_key: str
    mean_semantic_precision: float
    mean_topic_coverage: float
    mean_best_similarity: float
    mean_keyword_precision: float
    mean_latency_ms: float
    per_case: list
    num_cases: int


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SemanticEvaluator:
    """
    Evaluates retrieval quality using embedding similarity between
    retrieved chunks and expected topic descriptions.

    For each test case:
    1. Embed the query and retrieve top-5 chunks (same as keyword evaluator)
    2. Embed all expected topic descriptions
    3. Compute cosine similarity between each retrieved chunk and each topic
    4. Measure:
       - Semantic Precision: avg of max(similarity to any topic) for each chunk
       - Topic Coverage: fraction of topics where at least one chunk has sim > threshold
       - Best Similarity: highest chunk-topic similarity (sanity check)
    """

    RELEVANCE_THRESHOLD = 0.45  # chunk is "relevant" to a topic if sim > this

    def __init__(self):
        self.test_cases = SEMANTIC_TEST_CASES
        self.pipeline = EmbeddingPipeline()
        self.all_scores: list[SemanticConfigScore] = []

        # Load original keyword test cases for comparison
        kw_path = Path("evaluation/test_cases.json")
        if kw_path.exists():
            with open(kw_path, encoding="utf-8") as f:
                self.keyword_test_cases = json.load(f)
        else:
            self.keyword_test_cases = []

    def evaluate_config(self, model_key, strategy_key, result):
        """Evaluate one model x strategy configuration."""
        model_info = EMBEDDING_MODELS[model_key]
        model = SentenceTransformer(model_info["name"])

        per_case = []

        for tc in self.test_cases:
            query = tc["query"]
            topics = tc["expected_topics"]

            # Retrieve top-5
            start = time.perf_counter()
            query_emb = model.encode([query])
            sims = cosine_similarity(query_emb, result.embeddings)[0]
            top_indices = np.argsort(sims)[-5:][::-1]
            latency = (time.perf_counter() - start) * 1000

            top_chunks = [result.chunks[i].text for i in top_indices]

            # Embed topics and chunks
            topic_embs = model.encode(topics)
            chunk_embs = model.encode(top_chunks)

            # Similarity matrix: chunks (5) x topics (4)
            sim_matrix = cosine_similarity(chunk_embs, topic_embs)

            # Semantic Precision: for each chunk, how similar is it to its best-matching topic?
            chunk_best_sims = sim_matrix.max(axis=1)  # best topic match per chunk
            semantic_precision = float(np.mean(chunk_best_sims))

            # Topic Coverage: fraction of topics with at least one chunk above threshold
            topic_max_sims = sim_matrix.max(axis=0)  # best chunk match per topic
            topics_covered = sum(1 for s in topic_max_sims if s > self.RELEVANCE_THRESHOLD)
            topic_coverage = topics_covered / len(topics)

            # Best similarity (sanity check)
            best_sim = float(sim_matrix.max())

            # Original keyword metric for comparison
            kw_precision = 0.0
            kw_tc = next((k for k in self.keyword_test_cases if k["id"] == tc["id"]), None)
            if kw_tc:
                expected_kw = [kw.lower() for kw in kw_tc.get("expected_keywords", [])]
                if expected_kw:
                    combined = " ".join(top_chunks).lower()
                    kw_hits = sum(1 for kw in expected_kw if kw in combined)
                    kw_precision = kw_hits / len(expected_kw)

            per_case.append(SemanticScore(
                test_case_id=tc["id"],
                query=query[:80],
                model_key=model_key,
                strategy_key=strategy_key,
                semantic_precision=round(semantic_precision, 4),
                topic_coverage=round(topic_coverage, 4),
                best_chunk_similarity=round(best_sim, 4),
                keyword_precision=round(kw_precision, 4),
                latency_ms=round(latency, 2),
            ))

        config_score = SemanticConfigScore(
            model_key=model_key,
            strategy_key=strategy_key,
            mean_semantic_precision=round(float(np.mean([s.semantic_precision for s in per_case])), 4),
            mean_topic_coverage=round(float(np.mean([s.topic_coverage for s in per_case])), 4),
            mean_best_similarity=round(float(np.mean([s.best_chunk_similarity for s in per_case])), 4),
            mean_keyword_precision=round(float(np.mean([s.keyword_precision for s in per_case])), 4),
            mean_latency_ms=round(float(np.mean([s.latency_ms for s in per_case])), 2),
            per_case=per_case,
            num_cases=len(per_case),
        )

        logger.info(
            f"  {model_key} x {strategy_key}: "
            f"Sem-P={config_score.mean_semantic_precision:.3f}  "
            f"Top-Cov={config_score.mean_topic_coverage:.3f}  "
            f"KW-P={config_score.mean_keyword_precision:.3f}"
        )
        return config_score

    def run_full_evaluation(self, models=None, strategies=None):
        """Run all configurations."""
        models = models or list(EMBEDDING_MODELS.keys())
        strategies = strategies or list(CHUNKING_STRATEGIES.keys())

        self.pipeline.load_documents()

        for strategy_key in strategies:
            chunks = self.pipeline.chunk_documents(strategy_key)
            for model_key in models:
                result = self.pipeline.embed_chunks(chunks, model_key)
                score = self.evaluate_config(model_key, strategy_key, result)
                self.all_scores.append(score)

        return self.all_scores

    def print_summary(self):
        """Print ranked results."""
        ranked = sorted(self.all_scores, key=lambda x: x.mean_semantic_precision, reverse=True)

        print("\n" + "=" * 95)
        print("RAG SEMANTIC EVALUATION — FINAL RANKINGS")
        print("=" * 95)
        print(f"{'Rank':<6} {'Model':<15} {'Strategy':<22} {'Sem-Prec':>9} {'Top-Cov':>9} {'KW-Prec':>9} {'Latency':>10}")
        print("-" * 95)
        for i, cs in enumerate(ranked, 1):
            marker = " *" if i == 1 else ""
            print(
                f"{i:<6} {cs.model_key:<15} {cs.strategy_key:<22} "
                f"{cs.mean_semantic_precision:>9.3f} {cs.mean_topic_coverage:>9.3f} "
                f"{cs.mean_keyword_precision:>9.3f} {cs.mean_latency_ms:>8.1f}ms{marker}"
            )
        print("=" * 95)

        # Comparison insight
        if len(ranked) >= 2:
            best = ranked[0]
            print(f"\nBest config: {best.model_key} x {best.strategy_key}")
            print(f"  Semantic Precision: {best.mean_semantic_precision:.3f} "
                  f"(vs keyword P@5: {best.mean_keyword_precision:.3f})")
            print(f"  Topic Coverage: {best.mean_topic_coverage:.1%} of expected topics found")

    def export_results(self, path="evaluation/semantic_results.json"):
        """Export detailed results."""
        output = []
        for cs in self.all_scores:
            output.append({
                "model": cs.model_key,
                "strategy": cs.strategy_key,
                "semantic_precision": cs.mean_semantic_precision,
                "topic_coverage": cs.mean_topic_coverage,
                "best_similarity": cs.mean_best_similarity,
                "keyword_precision": cs.mean_keyword_precision,
                "latency_ms": cs.mean_latency_ms,
                "per_case": [
                    {
                        "id": s.test_case_id,
                        "semantic_precision": s.semantic_precision,
                        "topic_coverage": s.topic_coverage,
                        "keyword_precision": s.keyword_precision,
                    }
                    for s in cs.per_case
                ],
            })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic RAG evaluation")
    parser.add_argument("--model", choices=list(EMBEDDING_MODELS.keys()), help="Single model")
    parser.add_argument("--strategy", choices=list(CHUNKING_STRATEGIES.keys()), help="Single strategy")
    parser.add_argument("--compare", action="store_true", help="Show keyword vs semantic comparison")
    args = parser.parse_args()

    evaluator = SemanticEvaluator()

    models = [args.model] if args.model else None
    strategies = [args.strategy] if args.strategy else None

    evaluator.run_full_evaluation(models=models, strategies=strategies)
    evaluator.print_summary()
    evaluator.export_results()

    print("\nDone. Results in evaluation/semantic_results.json")


if __name__ == "__main__":
    main()
