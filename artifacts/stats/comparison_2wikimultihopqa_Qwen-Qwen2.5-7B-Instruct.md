# Baseline Comparison: Wo-RAG vs Single RAG

**Dataset:** `2wikimultihopqa`  
**Model:** `Qwen/Qwen2.5-7B-Instruct`  
**Split:** `validation`  
**Examples:** 200  
**Top-k (Single RAG):** 10  

## Summary

| Method | Avg EM | Avg F1 | Correct |
|--------|--------|--------|--------|
| Wo-RAG | 0.015 (1.5%) | 0.050 (5.0%) | 3/200 |
| Single RAG | 0.050 (5.0%) | 0.095 (9.5%) | 10/200 |

## Head-to-Head (EM)

- Single RAG wins: 7
- Wo-RAG wins: 0
- Ties: 193
- Avg EM delta (Single RAG − Wo-RAG): **+0.035**
- Avg F1 delta (Single RAG − Wo-RAG): **+0.044**
