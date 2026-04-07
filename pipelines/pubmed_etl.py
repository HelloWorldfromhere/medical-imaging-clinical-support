# =============================================================
# pipelines/pubmed_etl.py
# PubMed ETL Pipeline — Extract, Transform, Load
#
# PURPOSE: Automatically fetches medical abstracts from the
# PubMed public API and loads them into PostgreSQL.
#
# LEGAL NOTICE:
# - PubMed API is free and publicly available (NLM, NIH)
# - Email is required by NLM terms of service (included below)
# - Rate limits respected: max 3 requests/second without API key
# - No private or patient data (PHI) is used at any point
# - All data is cited by PMID as required
# =============================================================

import logging
import os
import time

import psycopg2
from Bio import Entrez
from dotenv import load_dotenv

# =============================================================
# CONFIGURATION — update your email below (required by PubMed)
# =============================================================
load_dotenv()

ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "medical_rag"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

# Medical topics to fetch — directly relevant to chest X-ray RAG system
SEARCH_TERMS = [
    "pneumonia chest xray diagnosis",
    "pneumothorax imaging radiology",
    "pulmonary infiltrates chest radiograph",
    "pleural effusion diagnosis imaging",
    "chest xray interpretation findings",
]

MAX_RESULTS_PER_TERM = 50   # 5 terms × 100 = 500 abstracts total

# =============================================================
# LOGGING SETUP
# =============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================
# STEP 1 — EXTRACT: Fetch abstracts from PubMed API
# =============================================================
def extract(search_term: str, max_results: int) -> list:
    """
    Fetch raw PubMed records for a given search term.
    Uses Biopython's Entrez interface (free, legal, public API).
    """
    logger.info(f"Extracting up to {max_results} records for: '{search_term}'")

    # Search for PMIDs
    search_handle = Entrez.esearch(
        db="pubmed",
        term=search_term,
        retmax=max_results,
        sort="relevance"
    )
    search_results = Entrez.read(search_handle)
    search_handle.close()

    pmids = search_results["IdList"]
    logger.info(f"  Found {len(pmids)} PMIDs")

    if not pmids:
        return []

    # Fetch full records for those PMIDs
    fetch_handle = Entrez.efetch(
        db="pubmed",
        id=pmids,
        rettype="abstract",
        retmode="xml"
    )
    records = Entrez.read(fetch_handle)
    fetch_handle.close()

    # Respect PubMed rate limit (3 requests/second without API key)
    time.sleep(1.0)

    return records.get("PubmedArticle", [])


# =============================================================
# STEP 2 — TRANSFORM: Clean and structure raw records
# =============================================================
def transform(raw_records: list) -> list:
    """
    Parse raw PubMed XML records into clean dictionaries
    ready for database insertion.
    """
    logger.info(f"Transforming {len(raw_records)} raw records...")

    transformed = []

    for record in raw_records:
        try:
            citation = record["MedlineCitation"]
            article  = citation["Article"]

            # --- PMID ---
            pmid = str(citation["PMID"])

            # --- Title ---
            title = str(article.get("ArticleTitle", "")).strip()
            if not title:
                continue   # Skip records with no title

            # --- Abstract ---
            abstract_data  = article.get("Abstract", {})
            abstract_parts = abstract_data.get("AbstractText", [])

            if isinstance(abstract_parts, list):
                abstract = " ".join([str(p) for p in abstract_parts]).strip()
            else:
                abstract = str(abstract_parts).strip()

            if not abstract or len(abstract) < 50:
                continue   # Skip records with no meaningful abstract

            # --- Publication date ---
            journal_issue = article.get("Journal", {}).get("JournalIssue", {})
            pub_date      = journal_issue.get("PubDate", {})
            year          = str(pub_date.get("Year", "2020"))
            publication_date = f"{year}-01-01"

            # --- Keywords ---
            keywords     = []
            keyword_list = citation.get("KeywordList", [])
            if keyword_list:
                keywords = [str(kw) for kw in keyword_list[0]]

            transformed.append({
                "pmid":             pmid,
                "title":            title,
                "abstract":         abstract,
                "publication_date": publication_date,
                "keywords":         keywords,
            })

        except Exception as e:
            logger.warning(f"  Skipping malformed record: {e}")
            continue

    logger.info(f"  Transformed {len(transformed)} valid records")
    return transformed


# =============================================================
# STEP 3 — LOAD: Insert records into PostgreSQL
# =============================================================
def load(transformed_data: list) -> dict:
    """
    Insert transformed records into the medical_documents table.
    Uses ON CONFLICT DO NOTHING to safely skip duplicates.
    """
    logger.info(f"Loading {len(transformed_data)} records into PostgreSQL...")

    inserted = 0
    skipped  = 0
    errors   = 0

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    for doc in transformed_data:
        try:
            cur.execute("""
                INSERT INTO medical_documents
                    (pmid, title, abstract, publication_date, keywords)
                VALUES
                    (%s, %s, %s, %s, %s)
                ON CONFLICT (pmid) DO NOTHING
            """, (
                doc["pmid"],
                doc["title"],
                doc["abstract"],
                doc["publication_date"],
                doc["keywords"],
            ))

            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1   # Already exists in DB

        except Exception as e:
            logger.error(f"  Error inserting PMID {doc.get('pmid')}: {e}")
            errors += 1
            continue

    conn.commit()
    cur.close()
    conn.close()

    result = {"inserted": inserted, "skipped": skipped, "errors": errors}
    logger.info(f"  Load complete: {result}")
    return result


# =============================================================
# FULL PIPELINE — run all search terms
# =============================================================
def run_pipeline():
    """
    Execute the full ETL pipeline for all medical search terms.
    Fetches from PubMed → transforms → loads into PostgreSQL.
    """
    Entrez.email = ENTREZ_EMAIL   # Required by NLM terms of service

    logger.info("=" * 60)
    logger.info("Starting PubMed ETL Pipeline")
    logger.info("=" * 60)

    total_inserted = 0
    total_skipped  = 0

    for term in SEARCH_TERMS:
        logger.info(f"\n--- Processing: '{term}' ---")

        # Extract
        raw_records = extract(term, MAX_RESULTS_PER_TERM)

        # Transform
        clean_records = transform(raw_records)

        # Load
        result = load(clean_records)

        total_inserted += result["inserted"]
        total_skipped  += result["skipped"]

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Total inserted : {total_inserted}")
    logger.info(f"  Total skipped  : {total_skipped} (duplicates)")
    logger.info("=" * 60)


# =============================================================
# VERIFY — check how many documents are in the database
# =============================================================
def verify():
    """
    Quick check: print how many documents are in medical_documents.
    Run this after the pipeline to confirm data loaded correctly.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM medical_documents;")
    count = cur.fetchone()[0]

    cur.execute("""
        SELECT publication_date, COUNT(*) as total
        FROM medical_documents
        GROUP BY publication_date
        ORDER BY total DESC
        LIMIT 5;
    """)
    breakdown = cur.fetchall()

    cur.close()
    conn.close()

    logger.info(f"\n Total documents in database: {count}")
    logger.info("Top publication years:")
    for row in breakdown:
        logger.info(f"  {row[0]} → {row[1]} articles")


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    run_pipeline()
    verify()
