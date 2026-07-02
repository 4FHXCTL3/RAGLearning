from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def require_api_key() -> str:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-siliconflow-api-key":
        raise RuntimeError(
            "未配置 OPENAI_API_KEY。请复制 .env.example 为 .env，"
            "并填入你的硅基流动 SiliconFlow API Key。"
        )
    return OPENAI_API_KEY
