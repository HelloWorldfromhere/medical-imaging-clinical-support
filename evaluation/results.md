# RAG Evaluation Results

**Evaluated:** 9 configurations (3 models × 3 chunking strategies)
**Test cases:** 20 clinical scenarios

## Summary Table

| Rank | Model | Chunking Strategy | Precision@5 | Keyword Coverage | Latency (ms) |
|------|-------|-------------------|-------------|-----------------|-------------|
| 1 | minilm | fixed_512 | **0.290** ⭐ | 0.200 | 14.5 |
| 2 | biolord | fixed_512 | **0.280** | 0.172 | 10.4 |
| 3 | biolord | recursive_paragraph | **0.280** | 0.152 | 12.2 |
| 4 | pubmedbert_st | fixed_512 | **0.270** | 0.182 | 11.1 |
| 5 | pubmedbert_st | recursive_paragraph | **0.250** | 0.193 | 56.8 |
| 6 | minilm | recursive_paragraph | **0.240** | 0.140 | 11.6 |
| 7 | minilm | sentence_based | **0.230** | 0.152 | 9.5 |
| 8 | biolord | sentence_based | **0.200** | 0.142 | 77.0 |
| 9 | pubmedbert_st | sentence_based | **0.160** | 0.113 | 72.7 |

## Recommended Configuration

**Model:** all-MiniLM-L6-v2 (General-purpose sentence transformer, 80 MB)
**Chunking:** fixed_512 — Fixed 512-character chunks, no overlap awareness

**Rationale:** Achieved highest Precision@5 (0.290) with 20.0% keyword coverage at 14.5ms average latency.

### Tradeoff Analysis

- vs runner-up (biolord × fixed_512): +0.010 precision, +4.1ms latency

### Performance by Difficulty

- Standard cases (n=10): P@5 = 0.260
- Complex cases (n=10): P@5 = 0.320