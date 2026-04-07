"""
Embedding Pipeline — Model & Chunking Comparison
Medical Imaging RAG Clinical Decision Support

Compares 3 embedding models x 3 chunking strategies = 9 configurations.
Results feed into ARCHITECTURE.md and evaluation/results.md.

Usage:
    python -m rag.embedding_pipeline              # Run full comparison
    python -m rag.embedding_pipeline --model biobert  # Single model
    python -m rag.embedding_pipeline --ingest      # Ingest best config into PostgreSQL
"""

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "description": "General-purpose sentence transformer, 80 MB",
        "dimension": 384,
    },
    "biolord": {
        "name": "FremyCompany/BioLORD-2023-M",
        "description": "Biomedical sentence transformer, 420 MB",
        "dimension": 768,
    },
    "pubmedbert_st": {
        "name": "pritamdeka/S-PubMedBert-MS-MARCO",
        "description": "PubMedBERT fine-tuned for semantic search, 420 MB",
        "dimension": 768,
    },
}

CHUNKING_STRATEGIES = {
    "fixed_512": {
        "description": "Fixed 512-character chunks, no overlap awareness",
        "splitter_class": CharacterTextSplitter,
        "params": {"chunk_size": 512, "chunk_overlap": 0, "separator": " "},
    },
    "recursive_paragraph": {
        "description": "Recursive splitting respecting paragraph > sentence > word boundaries",
        "splitter_class": RecursiveCharacterTextSplitter,
        "params": {
            "chunk_size": 450,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", ". ", " "],
        },
    },
    "sentence_based": {
        "description": "Sentence-level chunks, small granularity",
        "splitter_class": RecursiveCharacterTextSplitter,
        "params": {
            "chunk_size": 200,
            "chunk_overlap": 30,
            "separators": [". ", "! ", "? ", "\n"],
        },
    },
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    strategy: str
    char_count: int = 0
    content_hash: str = ""

    def __post_init__(self):
        self.char_count = len(self.text)
        self.content_hash = hashlib.md5(self.text.encode()).hexdigest()[:12]


@dataclass
class EmbeddingResult:
    model_key: str
    strategy_key: str
    num_chunks: int
    avg_chunk_size: float
    embed_time_sec: float
    avg_embed_time_ms: float
    dimension: int
    embeddings: Optional[np.ndarray] = field(default=None, repr=False)
    chunks: list = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    """
    Loads documents, chunks them with multiple strategies,
    embeds with multiple models, and stores comparison results.
    """

    def __init__(self, docs_path: str = "pipelines/pubmed_cache"):
        self.docs_path = Path(docs_path)
        self.documents: list[dict] = []
        self.results: list[EmbeddingResult] = []

    # ---- Document loading -------------------------------------------------

    def load_documents(self, source: str = "pubmed_json") -> list[dict]:
        """
        Load documents from PubMed ETL output or sample data.
        Returns list of dicts with 'id', 'title', 'abstract', 'full_text'.
        """
        json_path = self.docs_path / "documents.json"

        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                self.documents = json.load(f)
            logger.info(f"Loaded {len(self.documents)} documents from {json_path}")
        else:
            logger.warning(f"No documents at {json_path}, using sample medical corpus")
            self.documents = self._sample_medical_corpus()

        return self.documents

    @staticmethod
    def _sample_medical_corpus() -> list[dict]:
        """Fallback sample corpus for testing without PubMed data."""
        return [
            {
                "id": "PMID:29117159",
                "title": "Management of Community-Acquired Pneumonia in Adults",
                "abstract": "Community-acquired pneumonia remains a leading cause of hospitalization and death. Initial evaluation should include assessment of disease severity using validated scoring systems such as CURB-65 or the Pneumonia Severity Index. Empiric antibiotic therapy should cover typical and atypical pathogens. For outpatients, amoxicillin or doxycycline is recommended. For inpatients, combination therapy with a beta-lactam plus macrolide or respiratory fluoroquinolone monotherapy is preferred. Duration of therapy is typically 5-7 days.",
                "full_text": "",
            },
            {
                "id": "PMID:30272578",
                "title": "Diagnosis and Management of Pleural Effusion",
                "abstract": "Pleural effusion is a common clinical finding with diverse etiologies. Initial diagnostic evaluation involves thoracentesis with pleural fluid analysis. Light criteria distinguish transudative from exudative effusions based on protein and LDH ratios. Common transudative causes include heart failure and cirrhosis. Exudative effusions may result from infection, malignancy, or inflammatory conditions. CT imaging aids in identifying underlying pathology. Management depends on etiology and ranges from observation to chest tube drainage or pleurodesis.",
                "full_text": "",
            },
            {
                "id": "PMID:31613361",
                "title": "Acute Heart Failure: Diagnosis and Management",
                "abstract": "Acute decompensated heart failure presents with dyspnea, orthopnea, and bilateral pulmonary edema. Chest radiography shows cardiomegaly, cephalization of vessels, and bilateral infiltrates. BNP or NT-proBNP levels aid diagnosis. Immediate management includes IV loop diuretics such as furosemide, oxygen supplementation, and noninvasive ventilation with BiPAP or CPAP. Troponin should be measured to rule out acute coronary syndrome. Echocardiography is essential for assessing ejection fraction and guiding long-term therapy.",
                "full_text": "",
            },
            {
                "id": "PMID:28412303",
                "title": "Primary Spontaneous Pneumothorax Management",
                "abstract": "Primary spontaneous pneumothorax occurs predominantly in tall, thin young males without underlying lung disease. Small pneumothoraces less than 2cm on chest radiograph may be managed with observation and supplemental oxygen. Larger pneumothoraces or those with hemodynamic compromise require intervention. Needle aspiration is a first-line option for simple cases. Chest tube insertion is indicated for failed aspiration or large pneumothoraces. Recurrence rate is approximately 30% after the first episode. Video-assisted thoracoscopic surgery may be considered for recurrent cases.",
                "full_text": "",
            },
            {
                "id": "PMID:32293716",
                "title": "Interstitial Lung Disease: Classification and Approach",
                "abstract": "Interstitial lung diseases comprise a heterogeneous group of disorders affecting the lung parenchyma. High-resolution CT is the primary imaging modality showing patterns such as ground-glass opacities, reticular changes, honeycombing, and traction bronchiectasis. Idiopathic pulmonary fibrosis is the most common form presenting with progressive dyspnea and restrictive physiology on pulmonary function testing. Antifibrotic agents pirfenidone and nintedanib slow disease progression. Multidisciplinary discussion involving pulmonology, radiology, and pathology is recommended for diagnosis.",
                "full_text": "",
            },
            {
                "id": "PMID:28157742",
                "title": "Lung Cancer Screening and Diagnosis",
                "abstract": "Lung cancer is the leading cause of cancer death worldwide. Low-dose CT screening is recommended for adults aged 50-80 with 20 or more pack-year smoking history. Suspicious nodules larger than 8mm require further evaluation with PET-CT and tissue sampling. Staging uses the TNM classification and guides treatment decisions. Non-small cell lung cancer comprises approximately 85% of cases. Treatment options include surgical resection, stereotactic body radiation, chemotherapy, immunotherapy, and targeted therapy depending on stage and molecular profile.",
                "full_text": "",
            },
            {
                "id": "PMID:31558411",
                "title": "Tuberculosis: Diagnosis and Public Health Management",
                "abstract": "Tuberculosis caused by Mycobacterium tuberculosis remains a global health concern. Classic radiographic findings include upper lobe cavitary lesions, consolidation, and lymphadenopathy. Diagnosis requires sputum AFB smear microscopy, mycobacterial culture, and nucleic acid amplification testing such as GeneXpert MTB/RIF. Patients with suspected pulmonary TB should be placed in airborne isolation. Standard treatment involves a 6-month regimen of isoniazid, rifampin, pyrazinamide, and ethambutol for the initial 2 months, followed by isoniazid and rifampin for 4 months. Drug susceptibility testing guides adjustments for resistant strains.",
                "full_text": "",
            },
            {
                "id": "PMID:29461068",
                "title": "Post-operative Atelectasis: Prevention and Management",
                "abstract": "Atelectasis is the most common pulmonary complication following surgery, particularly abdominal and thoracic procedures. Risk factors include obesity, prolonged anesthesia, and pre-existing lung disease. Prevention strategies include incentive spirometry, early mobilization, and lung-protective ventilation during surgery. Treatment of established atelectasis involves deep breathing exercises, chest physiotherapy, and bronchodilators. Persistent atelectasis not responding to conservative measures may require bronchoscopy to evaluate for mucus plugging or endobronchial obstruction.",
                "full_text": "",
            },
            {
                "id": "PMID:30785925",
                "title": "Sarcoidosis: Clinical Features and Management",
                "abstract": "Sarcoidosis is a systemic granulomatous disease of unknown etiology commonly affecting the lungs and lymph nodes. Bilateral hilar lymphadenopathy on chest radiograph is the hallmark presentation. Serum ACE levels and calcium may be elevated. Diagnosis typically requires tissue biopsy showing noncaseating granulomas with exclusion of other causes. Many patients have spontaneous remission. Systemic corticosteroids are first-line therapy for progressive pulmonary disease, cardiac involvement, or neurologic manifestations. Steroid-sparing agents include methotrexate and azathioprine.",
                "full_text": "",
            },
            {
                "id": "PMID:31245813",
                "title": "Pneumocystis Pneumonia in Immunocompromised Patients",
                "abstract": "Pneumocystis jirovecii pneumonia is a life-threatening opportunistic infection occurring primarily in immunocompromised patients, particularly those with HIV and CD4 counts below 200 cells per microliter. Characteristic imaging shows bilateral diffuse ground-glass opacities. Diagnosis is confirmed by induced sputum or bronchoalveolar lavage with direct fluorescent antibody or PCR testing. First-line treatment is high-dose trimethoprim-sulfamethoxazole for 21 days. Adjunctive corticosteroids are indicated when PaO2 is below 70 mmHg. Prophylaxis with TMP-SMX is recommended for patients with CD4 below 200.",
                "full_text": "",
            },
            {
                "id": "PMID:29867094",
                "title": "Neonatal Respiratory Distress Syndrome",
                "abstract": "Neonatal respiratory distress syndrome results from surfactant deficiency primarily affecting premature infants born before 34 weeks gestation. Chest radiograph shows diffuse bilateral granular opacities with air bronchograms. Management includes exogenous surfactant administration, continuous positive airway pressure, and mechanical ventilation for severe cases. Antenatal corticosteroids given to mothers at risk of preterm delivery significantly reduce incidence and severity. Complications include bronchopulmonary dysplasia, intraventricular hemorrhage, and pneumothorax. Long-term outcomes have improved substantially with advances in neonatal intensive care.",
                "full_text": "",
            },
            {
                "id": "PMID:32185989",
                "title": "Traumatic Hemothorax: Assessment and Intervention",
                "abstract": "Traumatic hemothorax results from injury to chest wall vessels, intercostal arteries, or pulmonary parenchyma. Associated injuries include rib fractures, pulmonary contusion, and great vessel injury. Chest radiograph may initially underestimate blood volume as fluid layers posteriorly in supine position. CT thorax provides definitive assessment. Small hemothoraces may be observed. Chest tube thoracostomy is indicated for hemothoraces greater than 300mL or progressive accumulation. Massive hemothorax exceeding 1500mL or ongoing output greater than 200mL per hour warrants surgical exploration via thoracotomy.",
                "full_text": "",
            },
        ]

    # ---- Chunking ---------------------------------------------------------

    def chunk_documents(self, strategy_key: str) -> list[Chunk]:
        """Apply a chunking strategy to all loaded documents."""
        config = CHUNKING_STRATEGIES[strategy_key]
        splitter = config["splitter_class"](**config["params"])

        chunks = []
        for doc in self.documents:
            # Combine title + abstract (+ full_text if available)
            text = f"{doc['title']}\n\n{doc['abstract']}"
            if doc.get("full_text"):
                text += f"\n\n{doc['full_text']}"

            split_texts = splitter.split_text(text)
            for i, t in enumerate(split_texts):
                chunks.append(Chunk(
                    text=t,
                    doc_id=doc["id"],
                    chunk_index=i,
                    strategy=strategy_key,
                ))

        logger.info(
            f"Strategy '{strategy_key}': {len(chunks)} chunks from "
            f"{len(self.documents)} docs (avg {np.mean([c.char_count for c in chunks]):.0f} chars)"
        )
        return chunks

    # ---- Embedding --------------------------------------------------------

    def embed_chunks(self, chunks: list[Chunk], model_key: str) -> EmbeddingResult:
        """Embed all chunks with a given model and record timing."""
        model_info = EMBEDDING_MODELS[model_key]
        logger.info(f"Loading model: {model_info['name']} ...")

        model = SentenceTransformer(model_info["name"])
        texts = [c.text for c in chunks]

        start = time.perf_counter()
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        elapsed = time.perf_counter() - start

        result = EmbeddingResult(
            model_key=model_key,
            strategy_key=chunks[0].strategy if chunks else "unknown",
            num_chunks=len(chunks),
            avg_chunk_size=float(np.mean([c.char_count for c in chunks])),
            embed_time_sec=round(elapsed, 3),
            avg_embed_time_ms=round((elapsed / len(chunks)) * 1000, 2) if chunks else 0,
            dimension=embeddings.shape[1],
            embeddings=embeddings,
            chunks=chunks,
        )

        logger.info(
            f"  → {model_key} x {result.strategy_key}: "
            f"{result.num_chunks} chunks in {result.embed_time_sec}s "
            f"({result.avg_embed_time_ms} ms/chunk)"
        )
        return result

    # ---- Full comparison ---------------------------------------------------

    def run_comparison(self, models: list[str] | None = None, strategies: list[str] | None = None):
        """
        Run all model x strategy combinations and collect results.
        Returns list of EmbeddingResult (without large numpy arrays for printing).
        """
        models = models or list(EMBEDDING_MODELS.keys())
        strategies = strategies or list(CHUNKING_STRATEGIES.keys())

        self.load_documents()

        for strategy_key in strategies:
            chunks = self.chunk_documents(strategy_key)
            for model_key in models:
                result = self.embed_chunks(chunks, model_key)
                self.results.append(result)

        return self.results

    # ---- Results export ----------------------------------------------------

    def export_comparison_table(self, output_path: str = "evaluation/embedding_comparison.json"):
        """Export results as JSON for the evaluator to consume."""
        rows = []
        for r in self.results:
            rows.append({
                "model": r.model_key,
                "model_name": EMBEDDING_MODELS[r.model_key]["name"],
                "strategy": r.strategy_key,
                "strategy_description": CHUNKING_STRATEGIES[r.strategy_key]["description"],
                "num_chunks": r.num_chunks,
                "avg_chunk_chars": round(r.avg_chunk_size, 1),
                "embed_time_sec": r.embed_time_sec,
                "avg_embed_time_ms": r.avg_embed_time_ms,
                "dimension": r.dimension,
            })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        logger.info(f"Comparison table saved to {output_path}")

        # Print summary
        print("\n" + "=" * 85)
        print("EMBEDDING COMPARISON RESULTS")
        print("=" * 85)
        print(f"{'Model':<15} {'Strategy':<22} {'Chunks':>7} {'Avg Chars':>10} {'Time (s)':>9} {'ms/chunk':>9}")
        print("-" * 85)
        for r in rows:
            print(
                f"{r['model']:<15} {r['strategy']:<22} "
                f"{r['num_chunks']:>7} {r['avg_chunk_chars']:>10.1f} "
                f"{r['embed_time_sec']:>9.3f} {r['avg_embed_time_ms']:>9.2f}"
            )
        print("=" * 85)

        return rows

    # ---- PostgreSQL ingestion ----------------------------------------------

    def ingest_to_postgres(
        self,
        result: EmbeddingResult,
        db_config: dict | None = None,
    ):
        """
        Store the best configuration's embeddings into PostgreSQL with pgvector.
        Assumes schema from database/schema.sql is already applied.
        """
        db_config = db_config or {
            "host": "localhost",
            "port": 5432,
            "dbname": "medical_rag",
            "user": "postgres",
            "password": "",  # Set via .env
        }

        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Ensure pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create embeddings table if not exists
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                chunking_strategy TEXT NOT NULL,
                embedding vector({result.dimension}),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(content_hash, embedding_model)
            );
        """)

        # Batch insert
        rows = []
        for chunk, emb in zip(result.chunks, result.embeddings):
            rows.append((
                chunk.doc_id,
                chunk.chunk_index,
                chunk.text,
                chunk.content_hash,
                result.model_key,
                result.strategy_key,
                emb.tolist(),
            ))

        execute_values(
            cur,
            """
            INSERT INTO document_embeddings
                (doc_id, chunk_index, chunk_text, content_hash,
                 embedding_model, chunking_strategy, embedding)
            VALUES %s
            ON CONFLICT (content_hash, embedding_model) DO NOTHING
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s::vector)",
        )

        conn.commit()
        logger.info(f"Ingested {len(rows)} embeddings ({result.model_key} x {result.strategy_key}) into PostgreSQL")
        cur.close()
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embedding pipeline — model & chunking comparison")
    parser.add_argument("--model", choices=list(EMBEDDING_MODELS.keys()), help="Run single model only")
    parser.add_argument("--strategy", choices=list(CHUNKING_STRATEGIES.keys()), help="Run single strategy only")
    parser.add_argument("--ingest", action="store_true", help="Ingest best config into PostgreSQL")
    parser.add_argument("--docs-path", default="pipelines/pubmed_cache", help="Path to documents JSON")
    args = parser.parse_args()

    pipeline = EmbeddingPipeline(docs_path=args.docs_path)

    models = [args.model] if args.model else None
    strategies = [args.strategy] if args.strategy else None

    pipeline.run_comparison(models=models, strategies=strategies)
    pipeline.export_comparison_table()

    if args.ingest and pipeline.results:
        # Ingest the last result (or best — evaluator determines best)
        logger.info("Ingesting into PostgreSQL...")
        pipeline.ingest_to_postgres(pipeline.results[-1])


if __name__ == "__main__":
    main()
