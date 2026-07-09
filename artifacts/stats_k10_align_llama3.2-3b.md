# Evaluation Results

**测试模型 (LLM): `llama3.2:3b`**

- Model: `llama3.2:3b`
- Dataset split: `validation`
- Number of examples: 200
- Top-k retrieval: 10

## Summary Statistics

| Metric | Value |
|--------|-------|
| Average EM | **0.380** (38.0%) |
| Average F1 | **0.500** (50.0%) |

## Result Category

| Category | Count | Percentage |
|----------|-------|------------|
| Correct (EM > 0) | 76 | 38.0% |
| Rejected (unknown) | 0 | 0.0% |
| Incorrect answer | 124 | 62.0% |

## Retrieval Diagnosis (Step 0)

| Metric | Value |
|--------|-------|
| Avg Recall@10 (gold paragraphs) | **0.870** |
| Full-hit rate (all gold retrieved) | **0.755** |

**EM/F1 conditioned on retrieval** — separates "fix retrieval" from "fix generation":

| Condition | EM | F1 |
|-----------|----|----|
| Gold fully retrieved | 0.470 | 0.602 |
| Gold missed/partial | 0.102 | 0.185 |
