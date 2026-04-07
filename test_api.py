"""
Quick API test — run after starting the server.

Start server first:
    uvicorn api.main:app --reload --port 8000

Then in another terminal:
    python test_api.py
"""


import requests

BASE = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    print(f"GET /health: {r.status_code}")
    print(f"  {r.json()}")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    print("  PASS\n")


def test_stats():
    r = requests.get(f"{BASE}/stats")
    print(f"GET /stats: {r.status_code}")
    print(f"  {r.json()}")
    assert r.status_code == 200
    assert r.json()["corpus_size"] > 0
    print("  PASS\n")


def test_retrieve():
    payload = {
        "query": "65-year-old male with bilateral lobar consolidation on chest X-ray, history of COPD"
    }
    r = requests.post(f"{BASE}/retrieve", json=payload)
    print(f"POST /retrieve: {r.status_code}")
    data = r.json()
    print(f"  Chunks returned: {len(data['chunks'])}")
    print(f"  Retrieval latency: {data['retrieval_latency_ms']:.1f}ms")
    for i, chunk in enumerate(data["chunks"][:3]):
        print(f"  Chunk {i+1}: sim={chunk['similarity']:.3f}, {chunk['chunk_text'][:80]}...")
    assert r.status_code == 200
    assert len(data["chunks"]) > 0
    print("  PASS\n")


def test_query():
    payload = {
        "query": "65-year-old male with bilateral lobar consolidation, history of COPD",
        "cnn_prediction": "pneumonia",
        "confidence": 0.87,
    }
    r = requests.post(f"{BASE}/query", json=payload)
    print(f"POST /query: {r.status_code}")
    data = r.json()
    print(f"  Chunks: {len(data['chunks'])}")
    print(f"  Total latency: {data['total_latency_ms']:.1f}ms")
    print(f"  Response preview: {data['generated_response'][:100]}...")
    assert r.status_code == 200
    print("  PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("API TEST SUITE")
    print("=" * 60 + "\n")

    test_health()
    test_stats()
    test_retrieve()
    test_query()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
