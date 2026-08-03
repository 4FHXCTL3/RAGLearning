from __future__ import annotations

import ast
import html
from dataclasses import dataclass
from typing import Iterable

from rag_learning.config import HF_CACHE_DIR


# 2WikiMultihopQA parquet file URLs (one per split).
_TWO_WIKI_URLS = {
    "train": "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/train.parquet",
    "validation": "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet",
    "test": "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/test.parquet",
}

# MuSiQue config name (HuggingFace dataset `musique/musique`).
_MUSIQUE_CONFIG = "v2.0.1"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    example_id: str
    paragraph_id: int


def norm_title(title: str) -> str:
    """Normalize a Wikipedia title for gold/retrieved matching.

    Retrieved titles were html.unescaped when the index was built, so we do the
    same to gold supporting-fact titles and compare case-insensitively.
    """
    return html.unescape(title).strip().lower()


def gold_titles(example, dataset: str = "hotpotqa") -> set[str]:
    """Set of normalized gold supporting-paragraph titles for one example."""
    if dataset == "2wikimultihopqa":
        sf_raw = example.get("supporting_facts")
        if not sf_raw:
            return set()
        sf = ast.literal_eval(sf_raw) if isinstance(sf_raw, str) else sf_raw
        # sf is a list of [title, sentence_id] pairs
        return {norm_title(item[0]) for item in sf}
    if dataset == "musique":
        paragraphs = example.get("paragraphs", [])
        answer_paras = example.get("answer_paragraphs", [])
        titles = set()
        for idx in answer_paras:
            if idx < len(paragraphs):
                titles.add(norm_title(paragraphs[idx].get("title", "")))
        return titles
    # Default: HotpotQA
    sf = example.get("supporting_facts") or {}
    titles = sf.get("title", []) if isinstance(sf, dict) else []
    return {norm_title(t) for t in titles}


def load_examples(
    split: str = "validation", limit: int | None = None, dataset: str = "hotpotqa"
):
    """Load multi-hop QA examples.

    Parameters
    ----------
    split : str
        Dataset split, e.g. "validation" or "train".
    limit : int or None
        Truncate to this many examples (useful for quick experiments).
    dataset : str
        One of "hotpotqa" or "2wikimultihopqa".
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "无法导入 datasets。请先运行 `pip install -r requirements.txt`。"
        ) from exc

    if dataset == "2wikimultihopqa":
        url = _TWO_WIKI_URLS.get(split)
        if url is None:
            raise ValueError(
                f"Unknown split '{split}' for 2WikiMultihopQA. "
                f"Available: {list(_TWO_WIKI_URLS)}"
            )
        # Download the parquet file to local cache once, then read only the
        # row groups needed — avoids loading the full ~545 MB training set.
        import pyarrow.parquet as pq
        import pyarrow as pa
        import requests
        from datasets import Dataset
        from pathlib import Path

        cache_dir = Path(HF_CACHE_DIR) / "2wikimultihopqa"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / f"{split}.parquet"

        if not local_path.exists():
            print(f"Downloading {split}.parquet ...")
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(local_path, "wb") as f:
                if total:
                    from tqdm import tqdm
                    with tqdm(total=total, unit="B", unit_scale=True,
                              desc="Downloading") as pbar:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")

        # Read only enough row groups to cover `limit` rows.
        # Use iter_batches to avoid loading the entire single-row-group file
        # (~545 MB) into memory at once.
        pf = pq.ParquetFile(str(local_path))
        needed = limit if limit is not None else pf.metadata.num_rows
        rows_read = 0
        batches = []
        for batch in pf.iter_batches(batch_size=65536):
            rows_read += batch.num_rows
            batches.append(batch)
            if rows_read >= needed:
                break
        table = pa.Table.from_batches(batches)
        if limit is not None and rows_read > limit:
            table = table.slice(0, limit)
        ds = Dataset.from_list(table.to_pylist())
    elif dataset == "musique":
        split_str = f"{split}[:{limit}]" if limit else split
        ds = load_dataset("musique/musique", _MUSIQUE_CONFIG, split=split_str, cache_dir=HF_CACHE_DIR)
    else:
        split_str = f"{split}[:{limit}]" if limit else split
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_str, cache_dir=HF_CACHE_DIR)

    return ds


def iter_context_paragraphs(examples, dataset: str = "hotpotqa") -> Iterable[Document]:
    """Yield paragraph-level documents from dataset contexts."""
    for example_index, example in enumerate(examples):
        if dataset == "2wikimultihopqa":
            example_id = str(example.get("_id") or example_index)
            ctx_raw = example.get("context", "[]")
            context = ast.literal_eval(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
            # context is a list of [title, [sentence1, sentence2, ...]]
            for paragraph_id, (title, paragraph_sentences) in enumerate(context):
                text = " ".join(
                    sentence.strip()
                    for sentence in paragraph_sentences
                    if sentence.strip()
                )
                text = html.unescape(text)
                title = html.unescape(title)
                if not text:
                    continue
                yield Document(
                    id=f"{example_id}:{paragraph_id}",
                    title=title,
                    text=text,
                    example_id=example_id,
                    paragraph_id=paragraph_id,
                )
        elif dataset == "musique":
            example_id = str(example.get("id") or example_index)
            paragraphs = example.get("paragraphs", [])
            for paragraph_id, para in enumerate(paragraphs):
                title = html.unescape(para.get("title", ""))
                text = html.unescape(para.get("paragraph_text", ""))
                if not text:
                    continue
                yield Document(
                    id=f"{example_id}:{paragraph_id}",
                    title=title,
                    text=text,
                    example_id=example_id,
                    paragraph_id=paragraph_id,
                )
        else:
            # Default: HotpotQA
            example_id = str(example.get("id") or example_index)
            context = example["context"]
            titles = context["title"]
            sentences = context["sentences"]

            for paragraph_id, (title, paragraph_sentences) in enumerate(
                zip(titles, sentences)
            ):
                text = " ".join(
                    s.strip() for s in paragraph_sentences if s.strip()
                )
                text = html.unescape(text)
                title = html.unescape(title)
                if not text:
                    continue
                yield Document(
                    id=f"{example_id}:{paragraph_id}",
                    title=title,
                    text=text,
                    example_id=example_id,
                    paragraph_id=paragraph_id,
                )
