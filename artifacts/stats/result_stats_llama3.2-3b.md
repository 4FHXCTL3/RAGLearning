# Evaluation Results

**测试模型 (LLM): `llama3.2:3b`**

- Model: `llama3.2:3b`
- Dataset split: `validation`
- Number of examples: 200
- Top-k retrieval: 5

## Summary Statistics

| Metric | Value |
|--------|-------|
| Average EM | **0.075** (7.5%) |
| Average F1 | **0.133** (13.3%) |

## Result Category

| Category | Count | Percentage |
|----------|-------|------------|
| Correct (EM > 0) | 15 | 7.5% |
| Rejected (unknown) | 0 | 0.0% |
| Incorrect answer | 185 | 92.5% |

## Retrieval Diagnosis (Step 0)

| Metric | Value |
|--------|-------|
| Avg Recall@5 (gold paragraphs) | **0.645** |
| Full-hit rate (all gold retrieved) | **0.325** |

**EM/F1 conditioned on retrieval** — separates "fix retrieval" from "fix generation":

| Condition | EM | F1 |
|-----------|----|----|
| Gold fully retrieved | 0.092 | 0.209 |
| Gold missed/partial | 0.067 | 0.096 |
