"""
RAG Evaluator — Systematic retrieval quality measurement
Medical Imaging RAG Clinical Decision Support

Measures Precision@k for each embedding model × chunking strategy configuration
against the 20-case clinical test dataset.

Usage:
    python -m evaluation.rag_evaluator                    # Full 3x3 evaluation
    python -m evaluation.rag_evaluator --model biobert    # Single model
    python -m evaluation.rag_evaluator --export-md        # Generate markdown tables
"""

import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from rag.embedding_pipeline import (
    EmbeddingPipeline,
    EmbeddingResult,
    EMBEDDING_MODELS,
    CHUNKING_STRATEGIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievalScore:
    test_case_id: int
    query: str
    model_key: str
    strategy_key: str
    precision_at_5: float
    keyword_coverage: float
    top_5_chunks: list
    retrieval_latency_ms: float


@dataclass
class ConfigScore:
    model_key: str
    strategy_key: str
    mean_precision_at_5: float
    mean_keyword_coverage: float
    mean_latency_ms: float
    per_case_scores: list
    num_cases: int


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates retrieval quality across model × strategy configurations.

    Evaluation methodology:
    1. For each test case query, embed query with the target model
    2. Compute cosine similarity against all chunk embeddings
    3. Retrieve top-5 chunks
    4. Measure:
       - Precision@5: fraction of top-5 chunks containing expected keywords
       - Keyword coverage: fraction of expected keywords found in top-5 chunks
       - Latency: time from query embedding to ranked results
    """

    def __init__(self, test_cases_path: str = "evaluation/test_cases.json"):
        with open(test_cases_path) as f:
            self.test_cases = json.load(f)
        logger.info(f"Loaded {len(self.test_cases)} test cases")

        self.pipeline = EmbeddingPipeline()
        self.all_config_scores: list[ConfigScore] = []

    def evaluate_single_config(
        self,
        model_key: str,
        strategy_key: str,
        result: EmbeddingResult,
    ) -> ConfigScore:
        """Evaluate one model × strategy configuration against all test cases."""

        model = SentenceTransformer(EMBEDDING_MODELS[model_key]["name"])
        per_case: list[RetrievalScore] = []

        for tc in self.test_cases:
            query = tc["query"]
            expected_kw = [kw.lower() for kw in tc["expected_keywords"]]

            # Time the retrieval
            start = time.perf_counter()
            query_emb = model.encode([query])
            sims = cosine_similarity(query_emb, result.embeddings)[0]
            top_indices = np.argsort(sims)[-5:][::-1]
            latency = (time.perf_counter() - start) * 1000

            top_chunks = [result.chunks[i].text for i in top_indices]
            top_sims = [float(sims[i]) for i in top_indices]

            # Precision@5: how many of top-5 chunks contain at least one expected keyword
            relevant_count = 0
            for chunk_text in top_chunks:
                chunk_lower = chunk_text.lower()
                if any(kw in chunk_lower for kw in expected_kw):
                    relevant_count += 1
            precision_at_5 = relevant_count / 5

            # Keyword coverage: fraction of expected keywords found across all top-5 chunks
            combined_text = " ".join(top_chunks).lower()
            found_kw = sum(1 for kw in expected_kw if kw in combined_text)
            keyword_coverage = found_kw / len(expected_kw) if expected_kw else 0

            per_case.append(RetrievalScore(
                test_case_id=tc["id"],
                query=query,
                model_key=model_key,
                strategy_key=strategy_key,
                precision_at_5=precision_at_5,
                keyword_coverage=keyword_coverage,
                top_5_chunks=[
                    {"text": t[:120] + "...", "similarity": s}
                    for t, s in zip(top_chunks, top_sims)
                ],
                retrieval_latency_ms=round(latency, 2),
            ))

        config_score = ConfigScore(
            model_key=model_key,
            strategy_key=strategy_key,
            mean_precision_at_5=round(float(np.mean([s.precision_at_5 for s in per_case])), 4),
            mean_keyword_coverage=round(float(np.mean([s.keyword_coverage for s in per_case])), 4),
            mean_latency_ms=round(float(np.mean([s.retrieval_latency_ms for s in per_case])), 2),
            per_case_scores=per_case,
            num_cases=len(per_case),
        )

        logger.info(
            f"  {model_key} × {strategy_key}: "
            f"P@5={config_score.mean_precision_at_5:.3f}  "
            f"KW-Cov={config_score.mean_keyword_coverage:.3f}  "
            f"Latency={config_score.mean_latency_ms:.1f}ms"
        )
        return config_score

    def run_full_evaluation(
        self,
        models: list[str] | None = None,
        strategies: list[str] | None = None,
    ) -> list[ConfigScore]:
        """Run evaluation across all model × strategy combinations."""
        models = models or list(EMBEDDING_MODELS.keys())
        strategies = strategies or list(CHUNKING_STRATEGIES.keys())

        self.pipeline.load_documents()

        for strategy_key in strategies:
            chunks = self.pipeline.chunk_documents(strategy_key)
            for model_key in models:
                result = self.pipeline.embed_chunks(chunks, model_key)
                score = self.evaluate_single_config(model_key, strategy_key, result)
                self.all_config_scores.append(score)

        return self.all_config_scores

    # ---- Export ------------------------------------------------------------

    def export_results_json(self, path: str = "evaluation/results.json"):
        """Export detailed per-case results."""
        output = []
        for cs in self.all_config_scores:
            output.append({
                "model": cs.model_key,
                "strategy": cs.strategy_key,
                "mean_precision_at_5": cs.mean_precision_at_5,
                "mean_keyword_coverage": cs.mean_keyword_coverage,
                "mean_latency_ms": cs.mean_latency_ms,
                "num_cases": cs.num_cases,
                "per_case": [
                    {
                        "id": s.test_case_id,
                        "query": s.query[:80],
                        "precision_at_5": s.precision_at_5,
                        "keyword_coverage": s.keyword_coverage,
                        "latency_ms": s.retrieval_latency_ms,
                    }
                    for s in cs.per_case_scores
                ],
            })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {path}")

    def export_results_markdown(self, path: str = "evaluation/results.md") -> str:
        """Generate markdown comparison table for ARCHITECTURE.md."""

        # Sort by precision descending
        ranked = sorted(self.all_config_scores, key=lambda x: x.mean_precision_at_5, reverse=True)

        lines = [
            "# RAG Evaluation Results",
            "",
            f"**Evaluated:** {len(self.all_config_scores)} configurations "
            f"({len(set(c.model_key for c in ranked))} models × "
            f"{len(set(c.strategy_key for c in ranked))} chunking strategies)",
            f"**Test cases:** {ranked[0].num_cases if ranked else 0} clinical scenarios",
            "",
            "## Summary Table",
            "",
            "| Rank | Model | Chunking Strategy | Precision@5 | Keyword Coverage | Latency (ms) |",
            "|------|-------|-------------------|-------------|-----------------|-------------|",
        ]

        for i, cs in enumerate(ranked, 1):
            winner = " ⭐" if i == 1 else ""
            lines.append(
                f"| {i} | {cs.model_key} | {cs.strategy_key} | "
                f"**{cs.mean_precision_at_5:.3f}**{winner} | "
                f"{cs.mean_keyword_coverage:.3f} | {cs.mean_latency_ms:.1f} |"
            )

        # Best config analysis
        best = ranked[0] if ranked else None
        if best:
            lines += [
                "",
                "## Recommended Configuration",
                "",
                f"**Model:** {EMBEDDING_MODELS[best.model_key]['name']} ({EMBEDDING_MODELS[best.model_key]['description']})",
                f"**Chunking:** {best.strategy_key} — {CHUNKING_STRATEGIES[best.strategy_key]['description']}",
                "",
                f"**Rationale:** Achieved highest Precision@5 ({best.mean_precision_at_5:.3f}) with "
                f"{best.mean_keyword_coverage:.1%} keyword coverage at {best.mean_latency_ms:.1f}ms average latency.",
                "",
                "### Tradeoff Analysis",
                "",
            ]
            if len(ranked) >= 2:
                runner = ranked[1]
                p5_diff = best.mean_precision_at_5 - runner.mean_precision_at_5
                lat_diff = best.mean_latency_ms - runner.mean_latency_ms
                lines.append(
                    f"- vs runner-up ({runner.model_key} × {runner.strategy_key}): "
                    f"+{p5_diff:.3f} precision, {lat_diff:+.1f}ms latency"
                )

        # Difficulty breakdown for best config
        if best:
            standard = [s for s in best.per_case_scores
                        if any(tc["id"] == s.test_case_id and tc["difficulty"] == "standard"
                               for tc in self.test_cases)]
            complex_ = [s for s in best.per_case_scores
                        if any(tc["id"] == s.test_case_id and tc["difficulty"] == "complex"
                               for tc in self.test_cases)]

            if standard and complex_:
                lines += [
                    "",
                    "### Performance by Difficulty",
                    "",
                    f"- Standard cases (n={len(standard)}): "
                    f"P@5 = {np.mean([s.precision_at_5 for s in standard]):.3f}",
                    f"- Complex cases (n={len(complex_)}): "
                    f"P@5 = {np.mean([s.precision_at_5 for s in complex_]):.3f}",
                ]

        md_text = "\n".join(lines)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_text)
        logger.info(f"Markdown results saved to {path}")
        return md_text

    def print_summary(self):
        """Print a quick console summary."""
        ranked = sorted(self.all_config_scores, key=lambda x: x.mean_precision_at_5, reverse=True)
        print("\n" + "=" * 80)
        print("RAG EVALUATION — FINAL RANKINGS")
        print("=" * 80)
        print(f"{'Rank':<6} {'Model':<15} {'Strategy':<22} {'P@5':>8} {'KW-Cov':>8} {'Latency':>10}")
        print("-" * 80)
        for i, cs in enumerate(ranked, 1):
            marker = " ★" if i == 1 else ""
            print(
                f"{i:<6} {cs.model_key:<15} {cs.strategy_key:<22} "
                f"{cs.mean_precision_at_5:>8.3f} {cs.mean_keyword_coverage:>8.3f} "
                f"{cs.mean_latency_ms:>8.1f}ms{marker}"
            )
        print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG evaluation — model × chunking comparison")
    parser.add_argument("--model", choices=list(EMBEDDING_MODELS.keys()), help="Evaluate single model")
    parser.add_argument("--strategy", choices=list(CHUNKING_STRATEGIES.keys()), help="Evaluate single strategy")
    parser.add_argument("--export-md", action="store_true", help="Generate markdown results table")
    args = parser.parse_args()

    evaluator = RAGEvaluator()

    models = [args.model] if args.model else None
    strategies = [args.strategy] if args.strategy else None

    evaluator.run_full_evaluation(models=models, strategies=strategies)
    evaluator.print_summary()
    evaluator.export_results_json()

    if args.export_md:
        evaluator.export_results_markdown()


if __name__ == "__main__":
    main()
