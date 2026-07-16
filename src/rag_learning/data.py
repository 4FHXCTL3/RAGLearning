from __future__ import annotations

import ast
import html
from dataclasses import dataclass
from typing import Iterable


# 2WikiMultihopQA parquet file URLs (one per split).
_TWO_WIKI_URLS = {
    "train": "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/train.parquet",
    "validation": "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet",
    "test": "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/test.parquet",
}


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
        ds = load_dataset("parquet", data_files=url, split="train")
    else:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
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
