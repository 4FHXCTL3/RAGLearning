from pathlib import Path
import os
import re

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def model_slug(model: str | None = None) -> str:
    """Turn a model name into a filename-safe slug.

    Model names like ``llama3.2:3b`` or ``Qwen/Qwen2.5-7B-Instruct`` contain
    characters (``:`` ``/``) that are invalid in Windows filenames, so we map
    any non-alphanumeric run to a single dash. Used to name result files per
    model so different LLM runs don't overwrite each other.
    """
    name = model or LLM_MODEL
    return re.sub(r"[^0-9A-Za-z.]+", "-", name).strip("-")


def require_api_key() -> str:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-siliconflow-api-key":
        raise RuntimeError(
            "未配置 OPENAI_API_KEY。请在项目根目录的 .env 中设置 OPENAI_API_KEY，"
            "本地 Ollama 填占位值 ollama 即可；在线 API 则填对应平台的 Key。"
        )
    return OPENAI_API_KEY
