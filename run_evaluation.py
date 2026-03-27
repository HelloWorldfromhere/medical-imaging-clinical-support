#!/usr/bin/env python3
"""
Run Full RAG Evaluation
=======================
One command to run the complete 3-model × 3-strategy comparison.

Usage:
    python run_evaluation.py           # Full 3×3 evaluation
    python run_evaluation.py --quick   # Single best-guess config (biobert × recursive)

Output:
    evaluation/results.json            # Detailed per-case scores
    evaluation/results.md              # Markdown table for ARCHITECTURE.md
    evaluation/embedding_comparison.json  # Embedding timing data
"""

import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick run: biobert × recursive only")
    parser.add_argument("--model", help="Specific model: minilm, biobert, pubmedbert")
    parser.add_argument("--strategy", help="Specific strategy: fixed_512, recursive_paragraph, sentence_based")
    args = parser.parse_args()

    # Check dependencies
    try:
        import sentence_transformers
        import sklearn
        import numpy
        import langchain
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with: pip install sentence-transformers scikit-learn numpy langchain langchain-text-splitters")
        sys.exit(1)

    from evaluation.rag_evaluator import RAGEvaluator

    evaluator = RAGEvaluator(test_cases_path="evaluation/test_cases.json")

    if args.quick:
        models = ["biobert"]
        strategies = ["recursive_paragraph"]
        logger.info("Quick mode: biobert × recursive_paragraph only")
    elif args.model or args.strategy:
        models = [args.model] if args.model else None
        strategies = [args.strategy] if args.strategy else None
    else:
        models = None  # All 3
        strategies = None  # All 3
        logger.info("Full evaluation: 3 models × 3 strategies = 9 configurations")

    print("\n" + "=" * 60)
    print("MEDICAL RAG EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Models:     {models or 'ALL (minilm, biobert, pubmedbert)'}")
    print(f"Strategies: {strategies or 'ALL (fixed_512, recursive_paragraph, sentence_based)'}")
    print(f"Test cases: 20 clinical scenarios")
    print("=" * 60 + "\n")

    evaluator.run_full_evaluation(models=models, strategies=strategies)
    evaluator.print_summary()
    evaluator.export_results_json()
    evaluator.export_results_markdown()

    print("\n✅ Evaluation complete!")
    print("   → evaluation/results.json     (detailed scores)")
    print("   → evaluation/results.md        (markdown table — paste into ARCHITECTURE.md)")
    print("\nNext step: Copy results.md content into ARCHITECTURE.md Phase 3 section")


if __name__ == "__main__":
    main()
