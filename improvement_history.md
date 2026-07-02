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
