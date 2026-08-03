# Baseline Comparison: Wo-RAG vs Single RAG

**Dataset:** `hotpotqa`  
**Model:** `llama3.2:3b`  
**Split:** `validation`  
**Examples:** 200  
**Top-k (Single RAG):** 10  

## Summary

| Method | Avg EM | Avg F1 | Correct |
|--------|--------|--------|--------|
| Wo-RAG | 0.055 (5.5%) | 0.107 (10.7%) | 11/200 |
| Single RAG | 0.080 (8.0%) | 0.099 (9.9%) | 16/200 |

## Head-to-Head (EM)

- Single RAG wins: 8
- Wo-RAG wins: 3
- Ties: 189
- Avg EM delta (Single RAG − Wo-RAG): **+0.025**
- Avg F1 delta (Single RAG − Wo-RAG): **-0.008**
