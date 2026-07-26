# Evaluation Results

**测试模型 (LLM): `llama3.2:3b`**

- Model: `llama3.2:3b`
- Dataset split: `validation`
- Number of examples: 200
- Top-k retrieval: 5

## Summary Statistics

| Metric | Value |
|--------|-------|
| Average EM | **0.360** (36.0%) |
| Average F1 | **0.471** (47.1%) |

## Result Category

| Category | Count | Percentage |
|----------|-------|------------|
| Correct (EM > 0) | 72 | 36.0% |
| Rejected (unknown) | 0 | 0.0% |
| Incorrect answer | 128 | 64.0% |

## Retrieval Diagnosis (Step 0)

| Metric | Value |
|--------|-------|
| Avg Recall@5 (gold paragraphs) | **0.703** |
| Full-hit rate (all gold retrieved) | **0.465** |

**EM/F1 conditioned on retrieval** — separates "fix retrieval" from "fix generation":

| Condition | EM | F1 |
|-----------|----|----|
| Gold fully retrieved | 0.527 | 0.657 |
| Gold missed/partial | 0.215 | 0.310 |
