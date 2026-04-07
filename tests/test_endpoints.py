"""
Tests for every FastAPI endpoint.
Run with:  pytest tests/ -v
"""

import io

# ── GET endpoints ──────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_healthy(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["corpus_loaded"] is True
        assert data["corpus_size"] == 775
        assert data["chunk_count"] == 120
        assert data["cnn_model_loaded"] is True

class TestStats:
    def test_returns_stats(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["corpus_size"] == 775
        assert data["embedding_model"] == "minilm"
        assert "conditions" in data
        assert len(data["conditions"]) == 15  # 14 pathologies + Normal

class TestConditions:
    def test_returns_condition_list(self, client):
        r = client.get("/conditions")
        assert r.status_code == 200
        data = r.json()
        assert "Pneumonia" in data["conditions"]
        assert "Normal" in data["conditions"]
        assert data["model_loaded"] is True

class TestLandingPage:
    def test_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        # Landing page returns HTML, not JSON
        assert "text/html" in r.headers["content-type"]


# ── POST /retrieve ─────────────────────────────────────────────────────────

class TestRetrieve:
    def test_valid_query(self, client):
        r = client.post("/retrieve", json={"query": "bilateral lobar consolidation in elderly patient"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["similarity"] > 0
        assert "retrieval_latency_ms" in data

    def test_with_condition(self, client):
        r = client.post("/retrieve", json={"query": "chest X-ray findings", "condition": "Pneumonia"})
        assert r.status_code == 200

    def test_query_too_short(self, client):
        """Query must be at least 5 characters (Pydantic validation)."""
        r = client.post("/retrieve", json={"query": "hi"})
        assert r.status_code == 422  # Unprocessable Entity

    def test_empty_body(self, client):
        r = client.post("/retrieve", json={})
        assert r.status_code == 422


# ── POST /query ────────────────────────────────────────────────────────────

class TestQuery:
    def test_full_rag_pipeline(self, client):
        r = client.post("/query", json={
            "query": "patient with bilateral infiltrates and fever",
            "cnn_prediction": "Pneumonia",
            "confidence": 0.85,
        })
        assert r.status_code == 200
        data = r.json()
        assert "generated_response" in data
        assert len(data["generated_response"]) > 0
        assert data["cnn_prediction"] == "Pneumonia"
        assert data["confidence"] == 0.85
        assert len(data["chunks"]) > 0

    def test_defaults(self, client):
        """Should work with just a query — other fields have defaults."""
        r = client.post("/query", json={"query": "consolidation on chest radiograph"})
        assert r.status_code == 200
        data = r.json()
        assert data["cnn_prediction"] == "unknown"
        assert data["confidence"] == 0.0

    def test_confidence_out_of_range(self, client):
        r = client.post("/query", json={"query": "some valid query here", "confidence": 1.5})
        assert r.status_code == 422


# ── POST /predict ──────────────────────────────────────────────────────────

class TestPredict:
    def _fake_image(self):
        """Create a minimal valid PNG (1x1 pixel) for upload."""
        import struct
        import zlib
        # Minimal PNG: header + IHDR + IDAT + IEND
        def chunk(ctype, data):
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = zlib.compress(b"\x00\x00\x00\x00")
        idat = chunk(b"IDAT", raw)
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    def test_valid_image(self, client):
        img = self._fake_image()
        r = client.post("/predict", files={"file": ("xray.png", io.BytesIO(img), "image/png")})
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 3
        assert data["predictions"][0]["condition"] == "Pneumonia"
        assert data["model_loaded"] is True

    def test_non_image_rejected(self, client):
        r = client.post("/predict", files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")})
        assert r.status_code == 400

    def test_no_file(self, client):
        r = client.post("/predict")
        assert r.status_code == 422


# ── POST /analyze ──────────────────────────────────────────────────────────

class TestAnalyze:
    def _fake_image(self):
        import struct
        import zlib
        def chunk(ctype, data):
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = zlib.compress(b"\x00\x00\x00\x00")
        idat = chunk(b"IDAT", raw)
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    def test_full_pipeline(self, client):
        img = self._fake_image()
        r = client.post("/analyze", files={"file": ("xray.png", io.BytesIO(img), "image/png")})
        assert r.status_code == 200
        data = r.json()
        # CNN predictions present
        assert len(data["predictions"]) > 0
        # RAG chunks retrieved
        assert len(data["chunks"]) > 0
        # LLM summary generated
        assert len(data["generated_response"]) > 0
        # Latencies tracked
        assert data["inference_latency_ms"] >= 0
        assert data["retrieval_latency_ms"] >= 0
        assert data["total_latency_ms"] >= 0
        # Detected conditions extracted
        assert "Pneumonia" in data["detected_conditions"]

    def test_with_query_context(self, client):
        img = self._fake_image()
        r = client.post(
            "/analyze",
            files={"file": ("xray.png", io.BytesIO(img), "image/png")},
            data={"query": "65-year-old male, smoker, presenting with fever"},
        )
        assert r.status_code == 200
        assert "65-year-old" in r.json()["query"]
