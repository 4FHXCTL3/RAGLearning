# `run_eval.py` 提速记录

> 起点：用户用以下命令在本地 4090 + Ollama 跑 TriviaQA 评测太慢：
> ```
> python scripts/run_eval.py --dataset triviaqa --split validation --limit 2000 \
>   --model llama3:8b --retrieval-mode hybrid --max-retrievals 2 \
>   --index artifacts/indexes/bm25_index_triviaqa_train+validation_all.pkl \
>   --policy artifacts/policies/triviaqa_llama3-8b_policy.npz \
>   --workers 8 --print-every 20 --extend
> ```

## 关键背景

- TriviaQA 索引有 **783,508** 个文档（远超之前 HotpotQA 的 8k 规模）
- `--retrieval-mode hybrid` 走 `HybridIndex.search`，内部会调用 BM25 和 Dense 两个分支
- `--policy` 模式下，每个 episode 最多 5 次 LLM 调用（reset 时 1 次贪心 + 2 次采样；retrieve 后再 2 次采样）

## 已完成的优化（5 项 + 1 项回退）

### ✅ 1. `HybridIndex.search` 改 O(1) 查找（**最大头**）
- **文件**：`src/rag_learning/index.py`
- **问题**：原来 `bm25_ranked = [self.documents.index(d) for d, _ in bm25_hits]`，对每个候选 doc 在 78 万长度的 list 上做线性 `__eq__`，每次检索要做几万次 dataclass 字段全比较。
- **改法**：在 `HybridIndex.__init__` 和 `HybridIndex.load` 中建 `self._doc_id_to_idx: dict[str, int]`（用 `setdefault` 保留重复 id 的首次出现位置，与 `list.index()` 语义一致），`search` 里改成 `id2idx[d.id]`。
- **预期收益**：每题 CPU 时间下降明显（具体数字没测完，但每题 BM25+dense 两个分支各少做 60-120 次 O(n) 查找）。

### ✅ 2. 共享 Embedder
- **文件**：`scripts/run_eval.py`
- **问题**：`RAGEnv.__init__` 在 `embedder is None and retrieval_mode in {"dense","hybrid"}` 时会 `self.embedder = _Embedder()`，8 个 worker 各自创建独立 httpx client + 独立 cache。
- **改法**：`main()` 里在创建 env 之前 `shared_embedder = Embedder()`，传给每个 worker 的 `RAGEnv(embedder=shared_embedder, ...)`；`_evaluate_one` 也加了 `shared_embedder` 参数。

### ✅ 3. JSONL 写入改为缓冲
- **文件**：`scripts/run_eval.py`
- **问题**：每条记录都 `_append_record(records_path, record)`，里面 `open(..., "a") / write / close`。2000 道题 = 2000 次 fsync。
- **改法**：新增 `BufferedJSONLWriter`（context manager，64 条一批 flush），worker 路径用 `with BufferedJSONLWriter(records_path) as sidecar, ThreadPoolExecutor(...) as ex:` 包住。
- **保留**：旧 `_append_record` 函数还在，给 smoke test / 外部直接调用用。

### ✅ 4. 跳过末尾 JSONL 重复读
- **文件**：`scripts/run_eval.py`
- **问题**：`main()` 结尾会再调 `_load_existing_records(args.results_out, records_path)` 把整个 JSONL 从头读一遍 + parse + sort。
- **改法**：内存里 `combined = list(existing_records) + results`，按 `__idx` 排序一次。dedup 改成按 `example_id`（`--extend` 模式下，新一轮 `todo` 重新从 0 编号，与旧记录的 `__idx` 会重叠，所以必须用 `example_id`）。
- **顺带修复**：原来 `--extend` + `--print-every 1` 模式下，`new_idxs` 检查会把旧记录也打出来（重复打印），现在按 `example_id` 判断。

### ✅ 5. 提升内层 import
- **文件**：`scripts/run_eval.py`
- **问题**：`_evaluate_one` 里 `import numpy as np`、`from rag_learning.environment import RAGEnv` 等，每次调用都做 sys.modules 查找。
- **改法**：挪到模块顶部（在 `sys.path.insert` 之后）。

### ❌ 6. BM25 倒排索引（**回退**）
- **文件**：`src/rag_learning/index.py`（后已撤销）
- **尝试**：在 `BM25Index` 里加 `self._inverted: dict[term, list[(doc_idx, count)]]`，搜索时只对候选集评分（`rank_bm25.get_scores` 在 78 万文档上每查 2.4s，倒排能把候选降到几千）。
- **回退原因**：
  1. 倒排索引 build 时间过长。`doc_freqs` 是 783K 个 dict，总 posting ~30M，纯 Python `dict.setdefault + list.append` 跑 **2 分钟以上**，比每题节省的 0.9s × 2000 题 / 8 worker 还要多。
  2. 即使建好，对常见 query（"who wrote the novel ..."）的 IDF 过滤后候选集仍很大（因为很多词在多个 wiki entity 间重复），单次 search 只从 2.4s 降到 1.5s。
  3. 频繁跑 eval 时，每次 load 都要花这 2 分钟，单次性投入回报比极差。
- **如果想真做**：要么换成 `bm25s`（C++ 后端，单次 5-50ms）要么预 build 并 pickle 持久化（要改 `build_index.py`，要 rebuild 索引文件 ~3.8GB）。

---

## 还没做的优化方向

按收益从大到小排序：

### 🔴 大：`bm25s` 替换 `rank_bm25`
- `pip install bm25s[full]` 有 C++ 后端，10-100x 提速
- API 与 `rank_bm25.BM25Okapi` 接近，但需要适配 `tokenize` 规则
- 在 `BM25Index.__init__` 里加分支，已有索引可以保留 `rank_bm25` 路径兜底

### 🔴 大：预 build 倒排索引并持久化
- 在 `build_index.py` 里 build 时把 `_inverted` 一起 pickle 进去
- `HybridIndex.load` 时直接 unpickle，零额外 build 成本
- 风险：pickle 体积大（~2-4GB），加载慢

### 🟡 中：policy 提前终止
- 当前 `policy_answer` 最多跑 `max_steps=6` 步；reset 时已经采样 3 个 answer，policy 第 0 步直接 ACTION_STOP 是合法路径
- 可以在 `policy_answer` 里检查 `argmax(action_probs) == ACTION_STOP` 且贪心 answer 置信度足够时直接 break

### 🟡 中：`align_to_context` 在 policy 路径跳过
- `RAGPipeline.answer` 用，policy 路径不用 → 已经天然不调用
- 但 `_evaluate_one` 里没有 align，对 policy 路径无影响

### 🟢 小：`BM25Index.search` 用 `argpartition` 替代 `argsort`
- 36ms → ~10ms，但 2.4s 整体下收益有限
- 主要瓶颈在 `get_scores` 而不是排序

### 🟢 小：Embedder cache 持久化
- 当前 cache 在 worker 间共享（因为 #2），但每个进程重启都重 embed 一次 question
- 可以加个 on-disk cache（key 是 question text hash）

### 🟢 小：JSONL 改用 msgpack 或 orjson
- 现在每行 json.dumps + 一次 write，I/O 已经很轻（buffered）
- 改收益小

---

## 一些技术细节备忘

### TriviaQA 索引里的重复 id
- 78 万文档中 67 万是 unique id，11 万文档的 id 跟前面重复
- 原因：TriviaQA `entity_pages` 同一个段落会挂在多个 entity 下（`example_id:paragraph_id` 撞了）
- 改用 `dict` 查找时必须用 `setdefault` 而非 `dict[i] = i`（否则会去到最后一个匹配，跟原 `list.index()` 返回首个不一致）

### `--extend` 的 `__idx` 编号坑
- 新一轮 `todo = [(i, ex) for i, ex in enumerate(examples) if ex.get("_id") not in existing_ids]`
- `i` 永远是 0..len(todo)-1
- 旧记录的 `__idx` 是 0..N（N=已完成数）
- 所以**新旧 `__idx` 必然重叠**
- dedup / 打印必须用 `example_id` 不用 `__idx`

### 内存
- Embeddings 矩阵：783508 × 768 × 4 = 2.4GB
- tokenized_corpus 字符串：~500MB
- BM25Okapi doc_freqs：~500MB
- documents list：~1.5GB（含 title + text 字符串）
- 单进程 load 后约 5-6GB（系统 1TB 内存，不紧张）
- 8 worker 共享同一 index 不会复制数据（Python 引用）

### Ollama 配置
- `OLLAMA_NUM_PARALLEL=8`（见 `start-ollama.sh`）
- 跟 `--workers 8` 正好匹配；再加 worker 不会更快反而会争抢
- llama3:8b 在 4090 上约 50 token/s/stream × 8 parallel ≈ 400 token/s 聚合

---

## 重新跑基线测试的命令

```bash
cd /root/rivermind-data/CLARITY-RAG

# 烟囱测试（不污染 results）：
python scripts/run_eval.py \
  --dataset triviaqa --split validation --limit 20 --model llama3:8b \
  --retrieval-mode hybrid --max-retrievals 2 \
  --index artifacts/indexes/bm25_index_triviaqa_train+validation_all.pkl \
  --policy artifacts/policies/triviaqa_llama3-8b_policy.npz \
  --workers 8 --print-every 5 --force
# 注意 --force 才会重跑 20 道；不加 --force 会因为已有 results 立刻返回

# 真实评测（2000 道 extend 模式）：
python scripts/run_eval.py \
  --dataset triviaqa --split validation --limit 2000 --model llama3:8b \
  --retrieval-mode hybrid --max-retrievals 2 \
  --index artifacts/indexes/bm25_index_triviaqa_train+validation_all.pkl \
  --policy artifacts/policies/triviaqa_llama3-8b_policy.npz \
  --workers 8 --print-every 20 --extend
```

测时建议先看 `--print-every 5` 的进度输出，确认 tqdm 的 ETA 跟预期对得上。

---

## 文件改动清单

| 文件 | 改动 |
|---|---|
| `src/rag_learning/index.py` | `HybridIndex.__init__` / `.load` 加 `_doc_id_to_idx` 字典；`search` 用 dict 替代 `documents.index()` |
| `scripts/run_eval.py` | 提升内层 import；`main()` 创建 `shared_embedder`；新增 `BufferedJSONLWriter`；worker 路径用 buffered writer；末尾 `combined` 排序代替二次 JSONL 读；print 路径按 `example_id` 判断 |

回退的代码已完全删除，不留死分支。
