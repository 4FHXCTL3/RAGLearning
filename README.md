# HotpotQA 基础 RAG 项目

这是一个最小可运行的 RAG（Retrieval-Augmented Generation，检索增强生成）项目框架，用于后续实验和改进。当前版本使用：

- 数据集：HotpotQA `distractor` 配置；
- 检索器：BM25，先作为简单 baseline；
- LLM：通过硅基流动 SiliconFlow 的 OpenAI-compatible API 调用 Qwen 等在线模型；
- 评估：简化版 Exact Match 与 token F1。

> 本项目不会下载完整大模型。是否有免费 API 额度取决于硅基流动平台当前政策和你选择的模型。

## 1. 项目结构

```text
.
├── .env.example                 # SiliconFlow API 配置模板
├── README.md
├── requirements.txt
├── artifacts/                   # 保存 BM25 索引等产物
├── scripts/
│   ├── build_index.py           # 构建 HotpotQA BM25 索引
│   └── run_eval.py              # 运行小规模 RAG 验证
└── src/rag_learning/
    ├── config.py                # 路径和环境变量配置
    ├── data.py                  # HotpotQA 加载与段落文档转换
    ├── evaluate.py              # EM/F1 指标
    ├── index.py                 # BM25 索引
    ├── llm.py                   # SiliconFlow/OpenAI-compatible LLM 调用
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

## 3. 配置硅基流动 SiliconFlow API

1. 登录硅基流动平台，创建 API Key。
2. 复制环境变量模板：

   Windows PowerShell：

   ```powershell
   Copy-Item .env.example .env
   ```

   macOS / Linux / Git Bash：

   ```bash
   cp .env.example .env
   ```

3. 编辑 `.env`：

```env
OPENAI_API_KEY=你的硅基流动 API Key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

`LLM_MODEL` 请以硅基流动控制台或模型广场中你账号当前可调用的模型名为准。如果报“模型不存在”或“无权限”，就把这里替换成你能调用的免费/可用模型。

## 4. 构建 HotpotQA 检索索引

首次运行会通过 Hugging Face `datasets` 下载 HotpotQA 数据。先用很小的数据量跑通：

```bash
python scripts/build_index.py --split validation --limit 20
```

成功后会看到类似输出：

```text
Loaded 20 HotpotQA examples.
Indexed xxx context paragraphs.
Saved BM25 index to .../artifacts/bm25_index.pkl.
```

后续可以扩大规模：

```bash
python scripts/build_index.py --split validation --limit 200
python scripts/build_index.py --split train --limit 1000
```

## 5. 运行 RAG 验证

确认 `.env` 已经填好 API Key 后，运行：

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

### 这个项目会不会下载整个 Qwen 模型？

不会。LLM 通过 SiliconFlow API 调用，代码只发送 prompt 和接收答案。

### 为什么还会下载东西？

`datasets` 会下载 HotpotQA 数据集，这是数据集，不是大语言模型。

### 没有 API Key 能跑吗？

可以构建 BM25 索引；但运行 `run_eval.py` 调用 LLM 时必须配置 API Key。

### API 是完全免费的吗？

代码无法保证平台永久免费。请在硅基流动控制台查看当前免费额度、可免费调用的模型和计费规则。
