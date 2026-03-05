-- ============================================================
-- Medical Imaging Clinical Decision Support System
-- Database Schema
-- ============================================================
-- Legal Notice: This schema is for educational purposes only.
-- No private patient data (PHI) is stored at any point.
-- All data originates from public sources (NIH, PubMed).
-- ============================================================

-- Step 1: Enable pgvector extension (run this ONCE as postgres superuser)
-- This enables storing ML embeddings (384-dim vectors) directly in PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- TABLE 1: medical_documents
-- Stores PubMed medical abstracts ingested by the ETL pipeline.
-- The embedding_vector column holds BioBERT embeddings for RAG retrieval.
-- ============================================================
CREATE TABLE IF NOT EXISTS medical_documents (
    id                  SERIAL PRIMARY KEY,
    pmid                VARCHAR(20) UNIQUE NOT NULL,      -- PubMed ID (unique identifier)
    title               TEXT NOT NULL,                    -- Article title
    abstract            TEXT NOT NULL,                    -- Full abstract text
    publication_date    DATE,                             -- Publication date
    keywords            TEXT[],                           -- Array of keywords
    embedding_vector    VECTOR(384),                      -- BioBERT embedding (384 dimensions)
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);

-- Index for fast PMID lookups (used during ETL deduplication)
CREATE INDEX IF NOT EXISTS idx_medical_documents_pmid
    ON medical_documents(pmid);

-- Index for vector similarity search (used during RAG retrieval)
-- ivfflat is the standard pgvector index for cosine similarity
CREATE INDEX IF NOT EXISTS idx_medical_documents_embedding
    ON medical_documents USING ivfflat (embedding_vector vector_cosine_ops)
    WITH (lists = 100);

-- Index for keyword filtering (used in hybrid search)
CREATE INDEX IF NOT EXISTS idx_medical_documents_keywords
    ON medical_documents USING GIN (keywords);


-- ============================================================
-- TABLE 2: query_logs
-- Logs every API request for monitoring, analytics, and debugging.
-- This is what enables the /metrics endpoint in FastAPI.
-- ============================================================
CREATE TABLE IF NOT EXISTS query_logs (
    id                  SERIAL PRIMARY KEY,
    query_text          TEXT NOT NULL,                    -- The user's clinical query
    prediction          VARCHAR(100),                     -- CNN prediction (e.g. "pneumonia")
    confidence          FLOAT,                            -- CNN confidence score (0.0 - 1.0)
    retrieved_doc_ids   INTEGER[],                        -- Array of retrieved document IDs
    response_text       TEXT,                             -- LLM-generated clinical summary
    latency_ms          INTEGER,                          -- Total response time in milliseconds
    success             BOOLEAN DEFAULT TRUE,             -- Whether the request succeeded
    error_message       TEXT,                             -- Error details if success = FALSE
    created_at          TIMESTAMP DEFAULT NOW()
);

-- Index for time-based analytics queries (e.g. daily usage reports)
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at
    ON query_logs(created_at);

-- Index for filtering by prediction type (e.g. all pneumonia queries)
CREATE INDEX IF NOT EXISTS idx_query_logs_prediction
    ON query_logs(prediction);

-- Index for filtering failed requests quickly
CREATE INDEX IF NOT EXISTS idx_query_logs_success
    ON query_logs(success);


-- ============================================================
-- TABLE 3: model_versions
-- Tracks every trained CNN and embedding model.
-- Enables rollback, version comparison, and MLOps practices.
-- ============================================================
CREATE TABLE IF NOT EXISTS model_versions (
    id                  SERIAL PRIMARY KEY,
    model_type          VARCHAR(50) NOT NULL,             -- 'cnn' or 'embedding'
    version             VARCHAR(20) NOT NULL,             -- e.g. 'v1.0.3'
    accuracy            FLOAT,                            -- Validation accuracy
    metrics             JSONB,                            -- All metrics as JSON (precision, recall, F1, etc.)
    file_path           TEXT,                             -- Path to saved model checkpoint (.pth)
    is_active           BOOLEAN DEFAULT FALSE,            -- Only ONE model per type should be TRUE
    created_at          TIMESTAMP DEFAULT NOW()
);

-- Index for quickly finding the active model
CREATE INDEX IF NOT EXISTS idx_model_versions_active
    ON model_versions(model_type, is_active);

-- Index for version lookup
CREATE INDEX IF NOT EXISTS idx_model_versions_version
    ON model_versions(version);


-- ============================================================
-- SAMPLE ANALYTICS QUERIES
-- These are the SQL queries that power the /metrics endpoint.
-- Run these manually to verify your data looks correct.
-- ============================================================

-- Daily usage stats (last 7 days)
-- SELECT
--     DATE(created_at)        AS date,
--     COUNT(*)                AS total_queries,
--     AVG(latency_ms)         AS avg_latency_ms,
--     SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS success_rate
-- FROM query_logs
-- WHERE created_at > NOW() - INTERVAL '7 days'
-- GROUP BY DATE(created_at)
-- ORDER BY date DESC;

-- Most common CNN predictions
-- SELECT
--     prediction,
--     COUNT(*)                AS frequency,
--     AVG(confidence)         AS avg_confidence
-- FROM query_logs
-- WHERE prediction IS NOT NULL
-- GROUP BY prediction
-- ORDER BY frequency DESC;

-- Active model versions
-- SELECT model_type, version, accuracy, created_at
-- FROM model_versions
-- WHERE is_active = TRUE;
