from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    example_id: str
    paragraph_id: int


def load_examples(split: str = "validation", limit: int | None = None):
    """Load HotpotQA examples.

    The distractor setting is small enough for local experiments and contains
    the question, answer, and a list of context paragraphs for every example.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "无法导入 datasets。请先运行 `pip install -r requirements.txt`。"
        ) from exc

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def iter_context_paragraphs(examples) -> Iterable[Document]:
    """Yield paragraph-level documents from HotpotQA contexts."""
    for example_index, example in enumerate(examples):
        example_id = str(example.get("id") or example_index)
        context = example["context"]
        titles = context["title"]
        sentences = context["sentences"]

        for paragraph_id, (title, paragraph_sentences) in enumerate(zip(titles, sentences)):
            text = " ".join(sentence.strip() for sentence in paragraph_sentences if sentence.strip())
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
