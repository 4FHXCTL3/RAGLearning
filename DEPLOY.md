# 服务器部署（Ubuntu + NVIDIA GPU）

本地只做代码跑通验证（用已有的 `llama3.2:3b` 即可，**无需下载 4 个大模型**）；真正的
3 数据集 × 4 模型训练测试在服务器上跑。

## 1. 传代码

```bash
git clone <repo> /opt/rag_learning
# 或 rsync（排除本地环境/缓存/产物）
rsync -av --exclude .git --exclude .venv --exclude .hf_cache --exclude artifacts/ \
  ./ user@server:/opt/rag_learning/
```

> `.env`、`.hf_cache/`、`artifacts/*.pkl`（索引/策略权重）被 gitignore，需在服务器上创建。
> 可选：把本地 `artifacts/indexes/` 一起 rsync 过去，可跳过重新建索引。

## 2. Python 环境

```bash
cd /opt/rag_learning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl status ollama    # 确认服务运行
ollama pull llama2:7b-chat mistral:7b llama3:8b qwen3:8b nomic-embed-text
ollama ps                  # PROCESSOR 列为 GPU 即用上了显卡
```

- `nomic-embed-text` 是 `StateEncoder`（训练时的状态编码）需要的嵌入模型。
- 4 个 tag 对应：LLaMA2-7B-chat / Mistral-7B-v0.1 / LLaMA3-8B / Qwen3-8B。

## 4. `.env`（gitignore，需创建）

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3:8b     # 默认模型；每次运行用 --model 覆盖即可
```

## 5. 建索引（首次需联网下载数据集到 .hf_cache/，缓存就绪前不要设 HF_DATASETS_OFFLINE）

```bash
# TriviaQA 自带 entity_pages 语料
python scripts/build_index.py --dataset triviaqa --split validation --limit 200
# PopQA 的检索语料 = Simple English Wikipedia（约 200MB）
python scripts/build_index.py --dataset wikisimple --split train --limit 50000
# NQ-Swap 的噪声语料 = 2WikiMultihopQA validation（答案检索不到，测幻觉鲁棒性）
python scripts/build_index.py --dataset 2wikimultihopqa --split validation --limit 1000
```

## 6. 训练 + 评估（3 数据集 × 4 模型）

```bash
declare -A split=( [triviaqa]=validation [popqa]=test [nqswap]=test )
for ds in triviaqa popqa nqswap; do
  for m in llama2:7b-chat mistral:7b llama3:8b qwen3:8b; do
    slug=$(echo $m | tr ':/. ' '----')
    python scripts/train_policy.py --dataset $ds --model $m --split ${split[$ds]} \
      --limit 200 --epochs 3
    python scripts/run_eval.py --dataset $ds --model $m --split ${split[$ds]} \
      --limit 100 --top-k 10 \
      --policy artifacts/policies/${ds}_${slug}_policy.npz
  done
done
```

> ⚠️ 上面这个循环里训练和评估用的 `--split` 相同（都是 `${split[$ds]}`），意味着 **评估集是训练时见过的题**——结果含数据泄漏，只能算"训练集 EM/F1"，不是论文级 held-out 数字。如果要报真实泛化数字，用下面 §7 的标准 train/eval split 流程。

## 7. 标准 train/eval split（小训练、大评估、hybrid 检索）

适用场景：triviaqa、hotpotqa 等 HuggingFace 上有 `train` / `validation` 两个 disjoint split 的数据集。**训练用小集合（~200 题），评估用大集合（8000+ 题）+ 训好的 policy + dense + BM25 hybrid 检索**——这是论文里报 F1/EM 的标准范式。

```bash
DS=triviaqa
M=llama3:8b
SLUG=llama3-8b

# 1. 建 combined 索引（覆盖 train + validation 两个 split 的所有 entity_pages，
#    内含 BM25 + dense embeddings + RRF 融合。每 split 取 2000 题去重后
#    ~15-20k unique 段落，build 时间 2-3 分钟。
#    不加 --limit 会索引全部 87k+9.9k 题，需要 1-3 小时且生成巨大 .pkl。）
python scripts/build_index.py --dataset $DS --split train --split validation \
    --limit 2000 \
    --output artifacts/indexes/bm25_index_${DS}_train+validation.pkl \
    --retrieval-mode hybrid

# 2. 训练（小集合，train split）
python scripts/train_policy.py --dataset $DS --model $M --split train \
    --limit 200 --epochs 3 --lr 0.001 --group-size 3 \
    --retrieval-mode hybrid \
    --index artifacts/indexes/bm25_index_${DS}_train+validation.pkl
# 期望 ~75 分钟；policy 训完保存到 artifacts/policies/${DS}_${SLUG}_policy.npz

# 3. 评估（held-out，validation split，8000 题）
python scripts/run_eval.py --dataset $DS --model $M --split validation \
    --limit 8000 --retrieval-mode hybrid \
    --index artifacts/indexes/bm25_index_${DS}_train+validation.pkl \
    --policy artifacts/policies/${DS}_${SLUG}_policy.npz
# 期望 ~5 小时；run_eval.py 会自动检测"训练和评估 split 相同"并打印 leakage 警告
# （如果先 train 用了 train split、eval 用了 validation split，不会警告，因为这是正确的 held-out）

# 4. 看结果
python scripts/visualize_results.py \
    --results artifacts/eval_results/eval_results_${DS}_${SLUG}_policy.json
```

**关键 flag**：
- `build_index.py --split train --split validation`（重复 flag 合并多个 split 到一个索引）
- `--retrieval-mode {bm25, dense, hybrid}`：hybrid 是默认也是推荐；切到 bm25 可以做消融对比
- `run_eval.py` 的 leakage 警告：检测到 policy 训练时的 split 和当前 eval 的 split 相同时会 warn

**预期收益**（vs 纯 BM25）：hybrid 检索在 triviaqa 上 F1 区间从 0.40~0.55 提到 0.50~0.65，足够补偿 held-out 评估的精度损失。

单组合手动示例：

```bash
python scripts/train_policy.py --dataset triviaqa --model llama3:8b --split validation --limit 200 --epochs 3
python scripts/run_eval.py --dataset triviaqa --model llama3:8b --split validation --limit 100 --top-k 10 \
  --policy artifacts/policies/triviaqa_llama3-8b_policy.npz
```

结果自动写入 `artifacts/eval_results/eval_results_{dataset}_{model}.json`、策略权重
`artifacts/policies/{dataset}_{model}_policy.npz`、训练日志 `artifacts/train_logs/`。

### 断点续训（自动，无需参数）

`train_policy.py` 现在**默认自动续训**：只要 `artifacts/policies/{dataset}_{model}_policy.npz` +
同名 `.ckpt.json` 存在，重跑同一命令就会从上次断电/中断处**精确继续**（包括当时正在跑的那
一条问题），不会重头开始。已跑满 `--epochs` 会提示并直接退出，想继续就加大 `--epochs`。
要丢弃旧结果重头来，加 `--from-scratch`。

```bash
# 断电后重跑同一条命令即可续训（无需任何额外参数）
python scripts/train_policy.py --dataset triviaqa --model llama3:8b --split validation --limit 200 --epochs 3

# 想从零重训（覆盖旧权重）
python scripts/train_policy.py --dataset triviaqa --model llama3:8b --split validation --limit 200 --epochs 3 --from-scratch
```

注意：`train_policy.py` 会定期（`--save-every`，默认 50 题）原子化写 checkpoint，断电最多丢
`--save-every` 条问题的进度。

## 6b. 训练中实时监控 + 可视化

训练过程中在另一个终端实时查看（每 N 秒重画一次曲线，Ctrl-C 停止）：

```bash
python scripts/visualize_training.py --dataset triviaqa --model llama3:8b --watch 10
```

- 进度条已带 `R / EMA-R / obj / dW`：`dW` 是每步权重最大变化量——**这是判断"训练是否正常"
  的关键信号**（GRPO 的 objective 天然≈0，别看它）。`dW` 长期为 0 = 梯度没在流动。
- 每 `--save-every` 步会打印一次 `[Health ...]` 行，自动提示：权重冻结 / 学到太慢 /
  回报下滑 / retrieve 占比过低 / stop 占比过高（试试更小 `--lr`）。

训练/评估结束后的可视化（都不需要联网，只看 JSON）：

```bash
python scripts/visualize_results.py --dataset triviaqa --model llama3:8b   # EM/F1 直方图 + 统计
python scripts/visualize_training.py --dataset triviaqa --model llama3:8b  # 单次静态曲线
python scripts/compare_experiments.py --train   # 汇总全部 eval_results_*.json：矩阵表 + 柱状图 + 热力图 + 训练曲线
```

`compare_experiments.py` 会扫 `artifacts/eval_results/` 下所有新格式结果，输出
`artifacts/stats/experiment_matrix.md` 和 `artifacts/figures/` 下的图。

## 数据集 → split → 检索语料对照

| dataset | eval split | 检索语料（索引） |
|---|---|---|
| triviaqa | validation（test 无公开答案） | 自带 entity_pages |
| popqa | test | `wikisimple`（Simple English Wikipedia，共享索引） |
| nqswap | test | `2wikimultihopqa` validation（噪声池，答案检索不到） |
| hotpotqa / 2wikimultihopqa / musique | validation / train | 自带 context（不变） |

## 注意事项

- **本地验证**：`python scripts/run_eval.py --dataset triviaqa --split validation --limit 3 --model llama3.2:3b`
  （用本地已有的小模型跑通代码路径，不下载 4 个大模型）。
- **Simple 维基口径**：PopQA 绝对分数不与用完整 20GB 维基的论文直接可比；它测试的是
  "模型在可检索/不可检索混合问题上的自适应检索行为"。
- **NQ-Swap 预期**：检索命中率（full_hit_rate）≈ 0、EM 偏低是设计使然（答案故意检索不到），
  重点看模型会不会从噪声中幻觉作答 / 策略如何取舍。
- **重跑加速**：数据集已缓存后可 `export HF_DATASETS_OFFLINE=1` 或给 `train_policy.py` 加 `--offline`。
- **Ollama 内存**：7-8B 模型 Q4 量化约 5GB，4 个模型若同时常驻需注意显存；训练是顺序调用，无冲突。
