"""
PubMed JSON Fetcher — Lightweight document acquisition for RAG evaluation
Medical Imaging RAG Clinical Decision Support

Fetches abstracts from PubMed and saves as JSON for the embedding pipeline.
No PostgreSQL required — this is the development/evaluation path.

LEGAL:
- PubMed API is free and publicly available (NLM, NIH)
- Email required by NLM terms of service
- Rate limit: max 3 requests/second without API key
- All data is publicly available medical literature
- Documents cited by PMID

Usage:
    python -m pipelines.pubmed_fetch_json                 # Fetch all topics
    python -m pipelines.pubmed_fetch_json --max-per-term 20  # Smaller fetch
    python -m pipelines.pubmed_fetch_json --verify        # Check existing data
"""

import argparse
import json
import logging
import time
from pathlib import Path

from Bio import Entrez

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Required by NLM terms of service — replace with your email
ENTREZ_EMAIL = "your.email@concordia.ca"

# Output path — this is where the embedding pipeline looks
OUTPUT_PATH = Path("pipelines/pubmed_cache/documents.json")

# Search terms covering all 20 test case conditions
# Each term targets a specific clinical scenario in test_cases.json
SEARCH_TERMS = [
    # Pneumonia variants (test cases 1, 2, 11, 12, 15)
    "community acquired pneumonia chest radiograph diagnosis treatment",
    "hospital acquired pneumonia MRSA nosocomial imaging",
    "pneumocystis pneumonia immunocompromised HIV ground glass",
    "non resolving pneumonia bronchoscopy malignancy workup",

    # Pleural conditions (test cases 3, 13, 14)
    "pleural effusion thoracentesis Light criteria diagnosis",
    "traumatic hemothorax chest tube rib fractures management",
    "lupus pleuritis autoimmune pleural effusion SLE",

    # Cardiac (test cases 4, 20)
    "acute decompensated heart failure chest radiograph cardiomegaly BNP",

    # Pneumothorax (test cases 5, 17)
    "primary spontaneous pneumothorax management chest tube observation",
    "tension pneumothorax emergency needle decompression tracheal deviation",

    # Oncology (test case 6)
    "lung cancer screening chest radiograph PET CT staging biopsy",

    # Interstitial / fibrosis (test case 7)
    "interstitial lung disease pulmonary fibrosis HRCT classification",

    # Asthma (test case 8)
    "asthma exacerbation hyperinflation chest radiograph bronchodilator",

    # Sarcoidosis (test case 9)
    "sarcoidosis bilateral hilar lymphadenopathy diagnosis granulomatous",

    # Atelectasis (test cases 10, 18)
    "post operative atelectasis incentive spirometry prevention management",

    # Neonatal (test case 12)
    "neonatal respiratory distress syndrome surfactant premature",

    # Normal X-ray differential (test case 16)
    "normal chest radiograph persistent cough spirometry asthma GERD",

    # Tuberculosis (test case 19)
    "pulmonary tuberculosis cavitary lesion AFB diagnosis isolation",
]

MAX_PER_TERM = 15  # 18 terms × 15 = up to 270 abstracts (after dedup: ~150-200)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_pubmed(search_term: str, max_results: int) -> list[dict]:
    """Fetch abstracts from PubMed for a single search term."""
    logger.info(f"Searching: '{search_term}' (max {max_results})")

    # Step 1: Search for PMIDs
    search_handle = Entrez.esearch(
        db="pubmed", term=search_term, retmax=max_results, sort="relevance"
    )
    search_results = Entrez.read(search_handle)
    search_handle.close()

    pmids = search_results["IdList"]
    if not pmids:
        logger.warning("  No results found")
        return []

    logger.info(f"  Found {len(pmids)} PMIDs")

    # Step 2: Fetch full records
    fetch_handle = Entrez.efetch(
        db="pubmed", id=pmids, rettype="abstract", retmode="xml"
    )
    records = Entrez.read(fetch_handle)
    fetch_handle.close()

    # Respect rate limit (NLM requires max 3 requests/sec without API key)
    time.sleep(0.5)

    return records.get("PubmedArticle", [])


def parse_record(record: dict) -> dict | None:
    """Parse a single PubMed XML record into a clean dict."""
    try:
        citation = record["MedlineCitation"]
        article = citation["Article"]

        pmid = str(citation["PMID"])
        title = str(article.get("ArticleTitle", "")).strip()
        if not title:
            return None

        # Extract abstract text
        abstract_data = article.get("Abstract", {})
        abstract_parts = abstract_data.get("AbstractText", [])
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(p) for p in abstract_parts).strip()
        else:
            abstract = str(abstract_parts).strip()

        if not abstract or len(abstract) < 50:
            return None

        # Publication year
        pub_date = (
            article.get("Journal", {})
            .get("JournalIssue", {})
            .get("PubDate", {})
        )
        year = str(pub_date.get("Year", "unknown"))

        # Keywords
        keyword_list = citation.get("KeywordList", [])
        keywords = [str(kw) for kw in keyword_list[0]] if keyword_list else []

        return {
            "id": f"PMID:{pmid}",
            "title": title,
            "abstract": abstract,
            "full_text": "",  # PubMed only provides abstracts for free
            "year": year,
            "keywords": keywords,
        }
    except Exception as e:
        logger.warning(f"  Skipping malformed record: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_fetch(max_per_term: int = MAX_PER_TERM):
    """Fetch all topics and save deduplicated results to JSON."""
    Entrez.email = ENTREZ_EMAIL

    all_docs = {}  # keyed by PMID for deduplication

    for term in SEARCH_TERMS:
        raw_records = fetch_pubmed(term, max_per_term)
        for record in raw_records:
            doc = parse_record(record)
            if doc and doc["id"] not in all_docs:
                all_docs[doc["id"]] = doc

        logger.info(f"  Running total: {len(all_docs)} unique documents\n")

    # Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    docs_list = list(all_docs.values())

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(docs_list, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved {len(docs_list)} documents to {OUTPUT_PATH}")
    return docs_list


def verify():
    """Check what's currently in the cache."""
    if not OUTPUT_PATH.exists():
        logger.info("No cached documents found. Run the fetch first.")
        return

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    logger.info(f"\nCached documents: {len(docs)}")
    logger.info(f"Location: {OUTPUT_PATH}")

    # Show topic coverage
    all_text = " ".join(d["title"].lower() + " " + d["abstract"].lower() for d in docs)
    conditions = [
        "pneumonia", "effusion", "cardiomegaly", "pneumothorax",
        "fibrosis", "tuberculosis", "sarcoidosis", "atelectasis",
        "lung cancer", "asthma", "surfactant", "lupus",
    ]
    logger.info("\nCondition coverage:")
    for condition in conditions:
        count = all_text.count(condition)
        status = "✓" if count > 0 else "✗ MISSING"
        logger.info(f"  {condition:<20} mentions: {count:>4}  {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for RAG evaluation")
    parser.add_argument("--max-per-term", type=int, default=MAX_PER_TERM, help="Max results per search term")
    parser.add_argument("--verify", action="store_true", help="Check existing cached data")
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        run_fetch(max_per_term=args.max_per_term)
        verify()
