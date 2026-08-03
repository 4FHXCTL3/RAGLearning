# Baseline Comparison: Wo-RAG vs Single RAG

**Dataset:** `hotpotqa`  
**Model:** `Qwen/Qwen2.5-7B-Instruct`  
**Split:** `validation`  
**Examples:** 200  
**Top-k (Single RAG):** 10  

## Summary

| Method | Avg EM | Avg F1 | Correct |
|--------|--------|--------|--------|
| Wo-RAG | 0.035 (3.5%) | 0.056 (5.6%) | 7/200 |
| Single RAG | 0.035 (3.5%) | 0.038 (3.8%) | 7/200 |

## Head-to-Head (EM)

- Single RAG wins: 4
- Wo-RAG wins: 4
- Ties: 192
- Avg EM delta (Single RAG − Wo-RAG): **+0.000**
- Avg F1 delta (Single RAG − Wo-RAG): **-0.018**
