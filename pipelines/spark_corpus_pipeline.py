"""
PySpark Corpus Processing Pipeline — Batch ETL at Scale
Medical Imaging RAG Clinical Decision Support

Demonstrates production-grade data engineering:
  - PySpark DataFrame API with explicit schemas
  - Parallel chunking via Spark SQL explode + UDF
  - Content-hash deduplication via Spark SQL
  - mapPartitions for batched embedding generation
  - Parquet output (industry-standard columnar format)
  - Benchmark comparison against sequential pipeline

Usage:
    python -m pipelines.spark_corpus_pipeline                      # Full run
    python -m pipelines.spark_corpus_pipeline --benchmark          # Compare vs sequential
    python -m pipelines.spark_corpus_pipeline --scale-test         # Simulate 10K-100K docs
    python -m pipelines.spark_corpus_pipeline --partitions 8       # Custom parallelism

LEGAL:
    - Processes locally cached PubMed abstracts (already fetched via pubmed_fetch_json.py)
    - No external API calls — all data already on disk
    - PySpark runs locally (no cloud cluster costs)
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_PATH = Path("pipelines/pubmed_cache/documents.json")
OUTPUT_DIR = Path("pipelines/spark_output")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

# Schema for chunked output
CHUNK_SCHEMA = StructType([
    StructField("doc_id", StringType(), nullable=False),
    StructField("chunk_index", IntegerType(), nullable=False),
    StructField("chunk_text", StringType(), nullable=False),
    StructField("char_count", IntegerType(), nullable=False),
    StructField("content_hash", StringType(), nullable=False),
])


# ---------------------------------------------------------------------------
# Spark Session
# ---------------------------------------------------------------------------

def create_spark_session(app_name: str = "MedicalRAG-CorpusPipeline", partitions: int = 4) -> SparkSession:
    """
    Create a local PySpark session optimized for single-machine batch processing.

    Why these settings:
      - local[*] uses all available CPU cores
      - shuffle partitions set low for local mode (default 200 is for clusters)
      - 2g driver memory is sufficient for our corpus size
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.log.level", "WARN")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Stage 1: Ingest — Load raw documents into Spark DataFrame
# ---------------------------------------------------------------------------

def ingest_documents(spark: SparkSession, docs_path: Path, partitions: int = 4):
    """
    Read PubMed JSON corpus into a Spark DataFrame.

    Note: Our documents.json is a JSON array ([{...}, {...}]), not JSON Lines.
    Spark's read.json() expects one object per line, so we load with Python
    first and parallelize via createDataFrame. This is standard practice
    when working with non-JSONL sources.

    At scale, you'd convert to JSON Lines or Parquet as a preprocessing step.
    """
    logger.info(f"Stage 1: Ingesting documents from {docs_path}")
    start = time.perf_counter()

    # Load JSON array with Python (Spark expects JSON Lines, not arrays)
    with open(docs_path, encoding="utf-8") as f:
        documents = json.load(f)

    # Convert to list of Rows for Spark
    rows = []
    for doc in documents:
        rows.append((
            doc.get("id", ""),
            doc.get("title", ""),
            doc.get("abstract", ""),
            doc.get("full_text", ""),
            doc.get("year", ""),
        ))

    schema = StructType([
        StructField("id", StringType(), nullable=False),
        StructField("title", StringType(), nullable=False),
        StructField("abstract", StringType(), nullable=False),
        StructField("full_text", StringType(), nullable=True),
        StructField("year", StringType(), nullable=True),
    ])

    df = spark.createDataFrame(rows, schema=schema)
    df = df.repartition(partitions)

    count = df.count()
    elapsed = time.perf_counter() - start
    logger.info(f"  → Loaded {count} documents in {elapsed:.2f}s ({partitions} partitions)")

    return df


# ---------------------------------------------------------------------------
# Stage 2: Parse & Clean — Filter invalid records, normalize text
# ---------------------------------------------------------------------------

def clean_documents(df):
    """
    Filter and normalize documents using Spark SQL expressions.

    Why Spark:
      Column-level operations (trim, length, coalesce) execute as optimized
      JVM operations across partitions — no Python serialization overhead.
    """
    logger.info("Stage 2: Cleaning and validating documents")
    start = time.perf_counter()

    initial_count = df.count()

    df_clean = (
        df
        # Normalize whitespace
        .withColumn("title", F.trim(F.col("title")))
        .withColumn("abstract", F.trim(F.col("abstract")))
        .withColumn("full_text", F.coalesce(F.trim(F.col("full_text")), F.lit("")))
        # Combine into single text field for chunking
        .withColumn(
            "combined_text",
            F.concat_ws("\n\n", F.col("title"), F.col("abstract"), F.col("full_text"))
        )
        # Filter: must have title and abstract with minimum length
        .filter(F.length(F.col("title")) > 0)
        .filter(F.length(F.col("abstract")) >= 50)
        # Add text length for analysis
        .withColumn("text_length", F.length(F.col("combined_text")))
    )

    final_count = df_clean.count()
    elapsed = time.perf_counter() - start
    logger.info(f"  → {initial_count} → {final_count} documents ({initial_count - final_count} filtered) in {elapsed:.2f}s")

    return df_clean


# ---------------------------------------------------------------------------
# Stage 3: Chunk — Split documents using Spark SQL UDF
# ---------------------------------------------------------------------------

def chunk_documents(spark, df):
    """
    Apply RecursiveCharacterTextSplitter via a Spark SQL UDF.

    Why a UDF here instead of mapPartitions:
      - UDFs integrate cleanly with Spark's DataFrame API
      - The splitter is lightweight (no model loading), so per-row
        initialization overhead is negligible
      - Produces an array column we can explode — idiomatic Spark

    The chunking logic is identical to the sequential pipeline:
      RecursiveCharacterTextSplitter with separators respecting
      paragraph > sentence > word boundaries.
    """
    logger.info("Stage 3: Chunking documents with RecursiveCharacterTextSplitter")
    start = time.perf_counter()

    # Register chunking as a UDF that returns an array of strings
    def split_text_udf(text):
        """Split text into chunks — runs inside Spark workers."""
        if not text:
            return []
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
        return splitter.split_text(text)

    # Register UDF with Spark
    chunk_udf = F.udf(split_text_udf, ArrayType(StringType()))

    # Apply chunking: each document produces an array of chunks
    df_with_chunks = df.withColumn("chunks", chunk_udf(F.col("combined_text")))

    # Explode array into individual rows + add chunk index
    # posexplode gives both position (chunk_index) and value (chunk_text)
    df_exploded = (
        df_with_chunks
        .select(
            F.col("id").alias("doc_id"),
            F.posexplode(F.col("chunks")).alias("chunk_index", "chunk_text"),
        )
        .withColumn("char_count", F.length(F.col("chunk_text")))
        .withColumn("content_hash", F.md5(F.col("chunk_text")))
    )

    count = df_exploded.count()
    elapsed = time.perf_counter() - start
    logger.info(f"  → {count} chunks created in {elapsed:.2f}s")

    return df_exploded


# ---------------------------------------------------------------------------
# Stage 4: Deduplicate — Remove exact-match duplicate chunks
# ---------------------------------------------------------------------------

def deduplicate_chunks(df_chunks):
    """
    Content-hash deduplication using Spark SQL dropDuplicates.

    Why Spark:
      Deduplication across millions of rows is a natural fit for Spark's
      shuffle-based groupBy. At 775 docs duplicates are rare, but at
      scale (multiple fetch runs, overlapping search terms) this prevents
      embedding the same text twice — saving compute and storage.
    """
    logger.info("Stage 4: Deduplicating chunks by content hash")
    start = time.perf_counter()

    initial_count = df_chunks.count()

    # Keep first occurrence of each unique chunk
    df_deduped = df_chunks.dropDuplicates(["content_hash"])

    final_count = df_deduped.count()
    dupes = initial_count - final_count
    elapsed = time.perf_counter() - start
    logger.info(f"  → {initial_count} → {final_count} chunks ({dupes} duplicates removed) in {elapsed:.2f}s")

    return df_deduped


# ---------------------------------------------------------------------------
# Stage 5: Embed — Generate embeddings using mapPartitions
# ---------------------------------------------------------------------------

def embed_chunks(df_chunks):
    """
    Generate embeddings using mapPartitions with batched inference.

    Why mapPartitions for embeddings:
      - The SentenceTransformer model is loaded ONCE per partition
        (~500MB model, 4 partitions = 4 loads vs thousands with a UDF)
      - Chunks within a partition are batched for CPU inference
      - This is the standard pattern for ML model inference in Spark
        (used at companies like Spotify, Netflix, and banks)

    Trade-off:
      Each partition loads its own copy of the model into memory.
      4 partitions × ~500MB = ~2GB. Acceptable for local mode.
      On a cluster, you'd use broadcast variables or model servers.
    """
    logger.info(f"Stage 5: Generating embeddings with {EMBEDDING_MODEL_NAME}")
    start = time.perf_counter()

    embedding_schema = StructType([
        StructField("doc_id", StringType()),
        StructField("chunk_index", IntegerType()),
        StructField("chunk_text", StringType()),
        StructField("char_count", IntegerType()),
        StructField("content_hash", StringType()),
        StructField("embedding", ArrayType(FloatType())),
    ])

    def embed_partition(rows):
        """Load model once per partition, embed all chunks in batch."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        batch_rows = list(rows)

        if not batch_rows:
            return iter([])

        texts = [r.chunk_text for r in batch_rows]

        # Batch encode — much faster than one-at-a-time
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

        results = []
        for row, emb in zip(batch_rows, embeddings):
            results.append((
                row.doc_id,
                row.chunk_index,
                row.chunk_text,
                row.char_count,
                row.content_hash,
                [float(x) for x in emb],
            ))
        return iter(results)

    embedded_rdd = df_chunks.rdd.mapPartitions(embed_partition)
    df_embedded = embedded_rdd.toDF(embedding_schema)

    count = df_embedded.count()
    elapsed = time.perf_counter() - start
    logger.info(f"  → {count} chunks embedded in {elapsed:.2f}s")

    return df_embedded


# ---------------------------------------------------------------------------
# Stage 6: Write — Save to Parquet (industry-standard columnar format)
# ---------------------------------------------------------------------------

def write_parquet(df_embedded, output_dir: Path):
    """
    Write results to Parquet format.

    Why Parquet:
      - Columnar storage: read only the columns you need (e.g., just embeddings)
      - Compressed: ~3-5x smaller than JSON
      - Schema-preserving: types are embedded in the file
      - Industry standard: Spark, Pandas, DuckDB, BigQuery all read Parquet natively
      - This is what you'd write to in a real data pipeline before loading to a vector DB
    """
    logger.info(f"Stage 6: Writing to Parquet at {output_dir}")
    start = time.perf_counter()

    output_path = str(output_dir / "embedded_chunks")

    (
        df_embedded
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    elapsed = time.perf_counter() - start
    logger.info(f"  → Parquet written in {elapsed:.2f}s")

    return output_path


# ---------------------------------------------------------------------------
# Sequential baseline — for benchmark comparison
# ---------------------------------------------------------------------------

def run_sequential_baseline(docs_path: Path):
    """
    Process the same corpus sequentially (current approach) for timing comparison.
    Uses the same chunking and embedding logic, just without Spark.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    logger.info("=" * 60)
    logger.info("SEQUENTIAL BASELINE (current approach)")
    logger.info("=" * 60)

    total_start = time.perf_counter()

    # Load
    load_start = time.perf_counter()
    with open(docs_path, encoding="utf-8") as f:
        documents = json.load(f)
    load_ms = (time.perf_counter() - load_start) * 1000

    # Chunk
    chunk_start = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    seen_hashes = set()
    for doc in documents:
        text = f"{doc['title']}\n\n{doc['abstract']}"
        if doc.get("full_text"):
            text += f"\n\n{doc['full_text']}"
        for i, chunk_text in enumerate(splitter.split_text(text)):
            h = hashlib.md5(chunk_text.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                chunks.append({"doc_id": doc["id"], "chunk_index": i, "text": chunk_text})
    chunk_ms = (time.perf_counter() - chunk_start) * 1000

    # Embed
    embed_start = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    model.encode(texts, batch_size=64, show_progress_bar=True)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "method": "sequential",
        "docs": len(documents),
        "chunks": len(chunks),
        "load_ms": round(load_ms, 1),
        "chunk_ms": round(chunk_ms, 1),
        "embed_ms": round(embed_ms, 1),
        "total_ms": round(total_ms, 1),
    }


# ---------------------------------------------------------------------------
# Full Spark pipeline
# ---------------------------------------------------------------------------

def run_spark_pipeline(docs_path: Path, partitions: int = 4, skip_embeddings: bool = False):
    """
    Execute the full 6-stage Spark ETL pipeline.

    Stages:
      1. Ingest  — Load JSON into Spark DataFrame
      2. Clean   — Validate and normalize text
      3. Chunk   — RecursiveCharacterTextSplitter via Spark UDF + explode
      4. Dedup   — Content-hash deduplication
      5. Embed   — Batched sentence-transformer via mapPartitions
      6. Write   — Output to Parquet

    Returns timing dict for benchmark comparison.
    """
    logger.info("=" * 60)
    logger.info("SPARK PIPELINE")
    logger.info(f"  Partitions: {partitions}")
    logger.info(f"  Source: {docs_path}")
    logger.info("=" * 60)

    total_start = time.perf_counter()
    spark = create_spark_session(partitions=partitions)

    try:
        # Stage 1: Ingest
        ingest_start = time.perf_counter()
        df_docs = ingest_documents(spark, docs_path, partitions)
        ingest_ms = (time.perf_counter() - ingest_start) * 1000

        # Stage 2: Clean
        clean_start = time.perf_counter()
        df_clean = clean_documents(df_docs)
        clean_ms = (time.perf_counter() - clean_start) * 1000

        # Stage 3: Chunk
        chunk_start = time.perf_counter()
        df_chunks = chunk_documents(spark, df_clean)
        chunk_ms = (time.perf_counter() - chunk_start) * 1000

        # Stage 4: Deduplicate
        dedup_start = time.perf_counter()
        df_deduped = deduplicate_chunks(df_chunks)
        dedup_ms = (time.perf_counter() - dedup_start) * 1000

        doc_count = df_clean.count()
        chunk_count = df_deduped.count()

        if not skip_embeddings:
            # Stage 5: Embed
            embed_start = time.perf_counter()
            df_embedded = embed_chunks(df_deduped)
            embed_ms = (time.perf_counter() - embed_start) * 1000

            # Stage 6: Write
            write_start = time.perf_counter()
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            write_parquet(df_embedded, OUTPUT_DIR)
            write_ms = (time.perf_counter() - write_start) * 1000
        else:
            embed_ms = 0.0
            write_ms = 0.0
            logger.info("Skipping embedding and write stages (--skip-embeddings)")

        total_ms = (time.perf_counter() - total_start) * 1000

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SPARK PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Documents processed : {doc_count}")
        logger.info(f"  Chunks produced     : {chunk_count}")
        logger.info(f"  Partitions          : {partitions}")
        logger.info(f"  Ingest time         : {ingest_ms:>8.1f} ms")
        logger.info(f"  Clean time          : {clean_ms:>8.1f} ms")
        logger.info(f"  Chunk time          : {chunk_ms:>8.1f} ms")
        logger.info(f"  Dedup time          : {dedup_ms:>8.1f} ms")
        if not skip_embeddings:
            logger.info(f"  Embed time          : {embed_ms:>8.1f} ms")
            logger.info(f"  Write time          : {write_ms:>8.1f} ms")
        logger.info(f"  ─────────────────────────────────")
        logger.info(f"  Total               : {total_ms:>8.1f} ms")

        return {
            "method": "spark",
            "partitions": partitions,
            "docs": doc_count,
            "chunks": chunk_count,
            "ingest_ms": round(ingest_ms, 1),
            "clean_ms": round(clean_ms, 1),
            "chunk_ms": round(chunk_ms, 1),
            "dedup_ms": round(dedup_ms, 1),
            "embed_ms": round(embed_ms, 1),
            "write_ms": round(write_ms, 1),
            "total_ms": round(total_ms, 1),
        }

    finally:
        spark.stop()


# ---------------------------------------------------------------------------
# Scale test — simulate larger corpora
# ---------------------------------------------------------------------------

def run_scale_test(docs_path: Path, partitions: int = 4):
    """
    Simulate processing at 1x, 5x, 10x, 25x, 50x corpus sizes.
    Measures chunking + dedup throughput (skips embedding to isolate Spark perf).

    Why skip embedding in scale test:
      Embedding is bottlenecked by the sentence-transformer model (CPU/GPU bound),
      not by Spark. The scale test isolates what Spark actually accelerates:
      parallel text processing, chunking, and deduplication.
    """
    logger.info("=" * 60)
    logger.info("SCALE TEST — Chunking + Dedup Throughput")
    logger.info("=" * 60)

    with open(docs_path, encoding="utf-8") as f:
        base_docs = json.load(f)

    base_count = len(base_docs)
    scale_factors = [1, 5, 10, 25, 50]
    results = []

    for factor in scale_factors:
        # Create scaled dataset by duplicating with unique IDs
        scaled_docs = []
        for copy_num in range(factor):
            for doc in base_docs:
                new_doc = doc.copy()
                if copy_num > 0:
                    new_doc["id"] = f"{doc['id']}_copy{copy_num}"
                    # Slightly modify text so dedup doesn't remove everything
                    new_doc["abstract"] = f"[Batch {copy_num}] {doc['abstract']}"
                scaled_docs.append(new_doc)

        # Write temporary scaled file
        scaled_path = Path(f"pipelines/spark_output/_scaled_{factor}x.json")
        scaled_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaled_path, "w", encoding="utf-8") as f:
            json.dump(scaled_docs, f)

        doc_count = len(scaled_docs)
        logger.info(f"\n--- {factor}x scale: {doc_count} documents ---")

        result = run_spark_pipeline(scaled_path, partitions=partitions, skip_embeddings=True)
        result["scale_factor"] = factor
        result["base_docs"] = base_count
        results.append(result)

        # Clean up temp file
        scaled_path.unlink(missing_ok=True)

    # Print comparison table
    print("\n" + "=" * 75)
    print("SCALE TEST RESULTS — Spark Chunking + Dedup Throughput")
    print("=" * 75)
    print(f"{'Scale':>6} {'Docs':>8} {'Chunks':>8} {'Chunk ms':>10} {'Dedup ms':>10} {'Total ms':>10} {'docs/sec':>10}")
    print("-" * 75)
    for r in results:
        docs_per_sec = r["docs"] / (r["total_ms"] / 1000) if r["total_ms"] > 0 else 0
        print(
            f"{r['scale_factor']}x".rjust(6)
            + f"{r['docs']:>8}"
            + f"{r['chunks']:>8}"
            + f"{r['chunk_ms']:>10.1f}"
            + f"{r['dedup_ms']:>10.1f}"
            + f"{r['total_ms']:>10.1f}"
            + f"{docs_per_sec:>10.0f}"
        )
    print("=" * 75)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "scale_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Scale test results saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Benchmark — Spark vs Sequential
# ---------------------------------------------------------------------------

def run_benchmark(docs_path: Path, partitions: int = 4):
    """
    Head-to-head comparison: Spark pipeline vs sequential processing.
    Both process the same 775-doc corpus with identical chunking and embedding.
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: Spark vs Sequential")
    logger.info("=" * 60)

    # Run sequential first
    seq_result = run_sequential_baseline(docs_path)

    # Run Spark
    spark_result = run_spark_pipeline(docs_path, partitions=partitions)

    # Comparison
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: Spark vs Sequential")
    print("=" * 70)
    print(f"{'Metric':<25} {'Sequential':>15} {'Spark':>15} {'Speedup':>10}")
    print("-" * 70)

    metrics = [
        ("Documents", seq_result["docs"], spark_result["docs"], ""),
        ("Chunks", seq_result["chunks"], spark_result["chunks"], ""),
        ("Chunk time (ms)", seq_result["chunk_ms"], spark_result["chunk_ms"],
         f"{seq_result['chunk_ms'] / max(spark_result['chunk_ms'], 1):.1f}x"),
        ("Embed time (ms)", seq_result["embed_ms"], spark_result["embed_ms"],
         f"{seq_result['embed_ms'] / max(spark_result['embed_ms'], 1):.1f}x"),
        ("Total time (ms)", seq_result["total_ms"], spark_result["total_ms"],
         f"{seq_result['total_ms'] / max(spark_result['total_ms'], 1):.1f}x"),
    ]

    for label, seq_val, spark_val, speedup in metrics:
        if isinstance(seq_val, float):
            print(f"{label:<25} {seq_val:>15.1f} {spark_val:>15.1f} {speedup:>10}")
        else:
            print(f"{label:<25} {seq_val:>15} {spark_val:>15} {speedup:>10}")

    print("=" * 70)
    print("\nNOTE: At 775 documents, Spark overhead (JVM startup, serialization)")
    print("may make it slower than sequential. The advantage appears at scale.")
    print("Run --scale-test to see throughput scaling.\n")

    # Save benchmark
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_path = OUTPUT_DIR / "benchmark_results.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump({"sequential": seq_result, "spark": spark_result}, f, indent=2)
    logger.info(f"Benchmark results saved to {benchmark_path}")

    return seq_result, spark_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PySpark corpus processing pipeline with benchmarking"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run head-to-head comparison: Spark vs sequential",
    )
    parser.add_argument(
        "--scale-test", action="store_true",
        help="Simulate 1x-50x corpus sizes to show Spark scaling",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip embedding stage (faster iteration on pipeline logic)",
    )
    parser.add_argument(
        "--partitions", type=int, default=4,
        help="Number of Spark partitions (default: 4)",
    )
    parser.add_argument(
        "--docs-path", type=str, default=str(DOCS_PATH),
        help="Path to documents JSON",
    )
    args = parser.parse_args()

    docs = Path(args.docs_path)
    if not docs.exists():
        logger.error(f"Documents not found: {docs}")
        logger.error("Run 'python -m pipelines.pubmed_fetch_json' first.")
        exit(1)

    if args.benchmark:
        run_benchmark(docs, partitions=args.partitions)
    elif args.scale_test:
        run_scale_test(docs, partitions=args.partitions)
    else:
        run_spark_pipeline(docs, partitions=args.partitions, skip_embeddings=args.skip_embeddings)
