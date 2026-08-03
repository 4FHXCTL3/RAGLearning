# Baseline Comparison: Wo-RAG vs Single RAG

**Dataset:** `2wikimultihopqa`  
**Model:** `llama3.2:3b`  
**Split:** `validation`  
**Examples:** 200  
**Top-k (Single RAG):** 10  

## Summary

| Method | Avg EM | Avg F1 | Correct |
|--------|--------|--------|--------|
| Wo-RAG | 0.040 (4.0%) | 0.132 (13.2%) | 8/200 |
| Single RAG | 0.105 (10.5%) | 0.149 (14.9%) | 21/200 |

## Head-to-Head (EM)

- Single RAG wins: 17
- Wo-RAG wins: 4
- Ties: 179
- Avg EM delta (Single RAG − Wo-RAG): **+0.065**
- Avg F1 delta (Single RAG − Wo-RAG): **+0.017**
