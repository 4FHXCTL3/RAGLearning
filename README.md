# HotpotQA 基础 RAG 项目

这是一个最小可运行的 RAG（Retrieval-Augmented Generation，检索增强生成）项目框架，用于后续实验和改进。当前版本使用：

- 数据集：HotpotQA `distractor` 配置；
- 检索器：BM25，先作为简单 baseline；
- LLM：通过本地 Ollama 的 OpenAI-compatible API 调用 `llama3.2:3b`；
- 评估：简化版 Exact Match 与 token F1。

> LLM 在本地 Ollama 上运行，无需 API Key、无联网调用费用。代码仍走 OpenAI-compatible 接口，因此改回在线 API（如硅基流动 SiliconFlow）只需修改 `.env`。

## 1. 项目结构

```text
.
├── .env                         # LLM 配置（本地 Ollama / 在线 API）
├── README.md
├── requirements.txt
├── artifacts/                   # 实验结果与产物（索引、策略、评估等）
│   ├── indexes/                 # 检索索引文件（BM25、Dense）
│   ├── policies/                # 训练好的策略网络权重
│   ├── train_logs/              # GRPO 训练日志
│   ├── eval_results/            # 评估结果 JSON
│   ├── stats/                   # 统计报告 Markdown
│   └── figures/                 # 可视化图表
├── scripts/
│   ├── build_index.py           # 构建 HotpotQA BM25 索引
│   └── run_eval.py              # 运行小规模 RAG 验证
└── src/rag_learning/
    ├── config.py                # 路径和环境变量配置
    ├── data.py                  # HotpotQA 加载与段落文档转换
    ├── evaluate.py              # EM/F1 指标
    ├── index.py                 # BM25 索引
    ├── llm.py                   # OpenAI-compatible LLM 调用（本地地址自动绕过代理）
    └── rag.py                   # RAG 检索与回答流程
```

## 2. 准备 Python 环境

建议使用 Python 3.10 或更高版本。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果 PowerShell 不允许激活虚拟环境，可以临时执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux / Git Bash

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 配置本地 Ollama LLM

本项目默认用本地 [Ollama](https://ollama.com) 运行 `llama3.2:3b`。

1. 安装 Ollama 并拉取模型：

   ```bash
   ollama pull llama3.2:3b
   ```

   确认 Ollama 服务在运行（默认监听 `http://localhost:11434`）。

2. 在项目根目录创建 `.env` 文件（本地 Ollama 无需真实 API Key，占位即可）：

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2:3b
```

> **关于 VPN / 代理**：`llm.py` 会自动识别本地地址（`localhost` / `127.0.0.1`）并绕过系统代理，因此开着 VPN 下载数据集时，本地 LLM 调用也不会被代理拦截导致 502，无需手动开关 VPN。

### 改用在线 API（可选）

如需换成硅基流动 SiliconFlow 等在线服务，把 `.env` 改为：

```env
OPENAI_API_KEY=你的 API Key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

`LLM_MODEL` 以对应平台当前可调用的模型名为准。

## 4. 构建 HotpotQA 检索索引

首次运行会通过 Hugging Face `datasets` 下载 HotpotQA 数据。先用很小的数据量跑通：

```bash
python scripts/build_index.py --split validation --limit 20
```

成功后会看到类似输出：

```text
Loaded 20 HotpotQA examples.
Indexed xxx context paragraphs.
Saved BM25 index to .../artifacts/indexes/bm25_index.pkl.
```

后续可以扩大规模：

```bash
python scripts/build_index.py --split validation --limit 200
python scripts/build_index.py --split train --limit 1000
```

## 5. 运行 RAG 验证

确认 `.env` 已配好、Ollama 服务在运行后，运行：

```bash
python scripts/run_eval.py --split validation --limit 2 --top-k 5
```

脚本会对每条样例输出：

- 问题；
- HotpotQA 标准答案；
- RAG 预测答案；
- EM/F1；
- 检索到的 top-k 段落标题和分数。

小规模跑通后可以增加验证条数：

```bash
python scripts/run_eval.py --split validation --limit 10 --top-k 5
```

## 6. 后续可以改进的模块

这个项目故意保持简单，方便你逐步替换模块：

1. **检索器改进**：把 `src/rag_learning/index.py` 的 BM25 换成 embedding + FAISS，或做 BM25 + 向量混合检索。
2. **Reranker**：在 `src/rag_learning/rag.py` 中对 BM25 top-k 结果再排序。
3. **Prompt 优化**：修改 `SYSTEM_PROMPT` 和 `build_prompt`。
4. **评估增强**：在 `src/rag_learning/evaluate.py` 中加入 HotpotQA supporting facts、召回率、LLM judge 等指标。
5. **训练/微调**：基础版本不训练 LLM。后续可以添加检索器训练、reranker 训练或 LoRA 微调脚本。

## 7. 常见问题

### 这个项目会下载大模型吗？

会下载 LLM 权重，但由 Ollama 管理（`ollama pull llama3.2:3b`，约 2GB），而不是每次运行时下载。代码本身只通过本地 API 发送 prompt、接收答案。

### 为什么运行时还会下载东西？

`datasets` 会下载 HotpotQA 数据集，这是数据集，不是大语言模型。首次下载后会走本地缓存。

### 为什么开着 VPN 会报 502？

VPN 会设置系统代理，httpx 默认会把连本地 Ollama 的请求也走代理，导致 502。`llm.py` 已对本地地址自动绕过代理，所以现在开着 VPN 也能正常调用本地 LLM。

### 会用到 GPU 吗？

Ollama 会自动使用可用的 NVIDIA GPU。可用 `ollama ps` 查看 `PROCESSOR` 列确认是否为 GPU。

### 没有 GPU / 想换在线 API 能跑吗？

可以。Ollama 无 GPU 时会用 CPU（较慢）。也可按第 3 节改用在线 API。
