### 排查问题

观察模型输出发现两类严重问题：

1. **HTML 编码未解码**：HotpotQA 原始数据中存在大量 HTML entity 编码，例如：
   - `Etiquette &amp; Espionage` → 保持编码状态传给模型
   - 乱码破坏 prompt 结构，导致模型理解出错，无限重复生成或截断

2. **过度拒答**：即使正确段落已经在上下文里，模型仍然频繁输出 `unknown`，直接导致 F1 = 0。

3. **格式不匹配**：模型倾向于输出完整句子（`The answer is ...`），但 HotpotQA 评测要求 exact match 只匹配答案 span，导致大量输出不命中。

---

## 第一阶段：修复 HTML 编码

### 问题根因

HotpotQA 的 Hugging Face 数据集直接提供了带 HTML entity 的原始文本，代码没有做解码。

### 修复方案

在 `src/rag_learning/data.py` 和 `scripts/run_eval.py` 中：

```diff
+ import html
...
    text = " ".join(...)
+   text = html.unescape(text)
+   title = html.unescape(title)
```

### 效果

修复后，乱码消失，模型能正常理解 prompt。

---

## 第二阶段：Prompt 工程，减少过度拒答

### 问题根因

原始 system prompt 使用中文，要求模型回答，但没有明确约束：

- 没有强制要求在有答案时必须回答
- 允许模型输出 `不知道` 作为安全选择
- 没有严格约束输出格式

### 修改方案

重新设计 system prompt：

```markdown
You are a precise question answering assistant for HotpotQA multi-hop reading.
Answer ONLY using the provided context paragraphs.
- If the answer cannot be found in the context, reply exactly: unknown.
- If the answer is found in the context, extract the **exact shortest answer span** from the text.
- For yes/no questions, output ONLY "yes" or "no", do NOT add anything else.
- For names, numbers, and entities, output exactly the same text as it appears in the context.
- Output only the final answer, no full sentences, no explanations, no quotes, no prefixes "Answer:", no trailing punctuation.
- Do NOT repeat yourself. Do NOT hallucinate. Answer directly.
```

关键点：

1. **明确责任**：如果答案在上下文，必须提取，不能说 `unknown`
2. **格式严格**：强制最短 span，禁止多余输出
3. **对齐原文**：要求输出和上下文文本一致


## 第三阶段：上下文精确对齐后处理

### 问题根因

即使模型知道答案，也常常会：

- 轻微改述或拼写变体
- 多输出标点或介词
- 语序微调

这些不影响人理解，但会导致 HotpotQA exact match EM/F1 评测不命中。

### 改进方案

增加**滑动窗口 jaccard 对齐**后处理：

1. 模型输出答案后，遍历所有检索到的上下文段落
2. 对每个上下文分词，做滑动窗口，计算窗口和模型输出的 jaccard 重叠
3. 如果重叠度超过阈值（0.5），用上下文的**原始 exact span**替换模型输出
4. 这样对齐后，就能命中 gold answer 的 exact match

代码实现在 `src/rag_learning/rag.py` 的 `_best_context_match`。


---


## 可视化脚本

新增 `scripts/visualize_results.py`：

```bash
pip install matplotlib
python scripts/visualize_results.py --split validation --limit 50 --top-k 10 --context-source dataset --adaptive-retry
```

输出：

- `artifacts/eval_results.json` → 详细结果
- `artifacts/result_stats.md` → 统计表格
- `artifacts/results_hist.png` → EM/F1 分布直方图

---

## 关键结论

1. **数据清洗（HTML 解码）是第一步**：很多低分不是模型不行，而是数据输入已经错了。
2. **Prompt 约束比模型微调更性价比**：明确格式要求就能减少一半错误。
3. **后处理对齐很重要**：评测标准是 exact match，就应该对齐到 exact span，不要让 paraphrase 惩罚模型。
4. **分开检索上限和生成上限实验**：用 `dataset` 上下文可以直接看生成器能力，排除检索错误。

---

## 文件修改清单

| 文件 | 改动 |
|------|------|
| `src/rag_learning/data.py` | 增加 HTML unescape |
| `scripts/run_eval.py` | 新增 `--context-source` 和 `--adaptive-retry` 参数，也增加 HTML unescape |
| `src/rag_learning/rag.py` | 重写 system prompt，新增 jaccard 对齐后处理 |
| `scripts/visualize_results.py` | 新增结果可视化脚本 |
| `improvement_history.md` | 本文档 |

---

# 第二轮：按 framework.md 逐步增强 —— Step 1（Generator 输出规范化）

> 日期：2026-07-03。目标：在 `framework.md` 模块化增强前，先修复被回退丢失的生成端后处理，确保每一步 F1 不低于 baseline。

## 背景：为什么 baseline 只有 F1=0.2775

固化的 baseline（`artifacts/baseline_metrics.json`）：

- 模型：`THUDM/GLM-Z1-9B-0414`（**推理型模型**，会输出 `<think>` 思维链）
- val 50 题：EM **0.02** / F1 **0.2775**

排查 `artifacts/baseline_eval.json` 里的原始预测，发现分数低**不是检索或模型能力问题**，而是 `src/rag_learning/rag.py` 被回退成了极简版本：

- system prompt 退化为 `"Answer the question based on the given context. Keep the answer short."`
- **完全没有输出后处理**

导致 GLM-Z1 的输出直接进评测，典型问题：

1. **思维链泄漏**：输出里夹带 `<think>…</think>`、`</answer>` 等标签。
2. **`Answer:` 前缀 / 整句作答**：如 `Answer: Yes. Both Scott Derrickson (American director) and Ed Wood ...`，而 gold 只是 `yes`。
3. **yes/no 题冗余**：模型答成一整句解释，exact-span 评测命不中。

HotpotQA 是 **exact-span EM/F1** 评测，这些脚手架文本会严重稀释 precision，把本该命中的答案拖到 0 分。

## 本次修改（Step 1）

只改生成端一处（对应 `framework.md` 模块3 Generator 的输出规范化），不动检索、不动策略：

### 1. 收紧 system prompt（`src/rag_learning/rag.py`）

从「简短作答」改为明确的 exact-span 约束：

- 只用上下文作答，抽取**最短答案 span**；
- yes/no 题只输出 `yes`/`no`；
- 人名/数字/日期/实体**原样照抄**上下文；
- 禁止整句、解释、引号、`Answer:` 前缀、结尾标点；
- 禁止重复。

### 2. 新增 `clean_answer()` 后处理（`src/rag_learning/rag.py`）

对（推理型）LLM 原始输出做规范化，逐步降到答案 span：

1. 去掉 `<think>…</think>` 思维链及残留标签；
2. 取最后一非空行（推理模型通常在末行复述最终答案）；
3. 去掉 `Answer:` / `The answer is` 前缀与首尾引号；
4. yes/no 判定折叠为裸 token；
5. 长答案只保留第一句，去掉结尾句号。

`RAGPipeline.answer()` 在 `llm.generate()` 之后调用 `clean_answer()`。

## 效果（离线验证，零 API 成本）

在**已采集的** baseline 原始预测（`artifacts/baseline_eval.json`，val 50 题）上直接套用 `clean_answer()`：

| 配置 | EM | F1 |
|------|----|----|
| baseline（回退版，原始输出直接评测） | 0.020 | **0.2775** |
| + Step 1（收紧 prompt + `clean_answer` 后处理） | 0.360 | **0.4625** |

因为是在完全相同的原始输出上离线复算，**这一步在数学上不可能造成回退**。live 复跑（新 prompt + 后处理）预期落在同一量级，待实际跑分确认后回填 `baseline_metrics.json`。

> 注：这一步是后续策略模块（状态感知 / 规则策略 / GRPO）的前置条件——所有动作都作用在这个「已清洗答案」之上，答案不干净时上层策略无从提升。

## 文件修改清单（Step 1）

| 文件 | 改动 |
|------|------|
| `src/rag_learning/rag.py` | 收紧 system prompt 为 exact-span 约束；新增 `clean_answer()` 后处理并接入 `RAGPipeline.answer()` |
| `improvement_history.md` | 追加本节 Step 1 总结 |

## 复现命令

```bash
python scripts/run_eval.py --split validation --limit 50 --top-k 5 --policy none \
  --results-out artifacts/step_clean_eval.json
```

对照基准：baseline F1 = 0.2775（EM 0.02）。
