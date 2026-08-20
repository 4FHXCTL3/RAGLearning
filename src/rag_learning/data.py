from __future__ import annotations

import ast
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag_learning.config import ARTIFACTS_DIR, HF_CACHE_DIR


# Simple English Wikipedia config (used as the retrieval corpus behind PopQA,
# which ships no passages of its own). Small (~200MB) but covers common
# entities; long-tail questions will simply not retrieve — that mix is the
# intended test scenario.
# NOTE: use the parquet-backed "wikimedia/wikipedia" repo. datasets>=3.2 no
# longer supports script-based datasets, and the legacy "wikipedia" repo is
# script-only, so it raises "Dataset scripts are no longer supported".
WIKI_SIMPLE_REPO = "wikimedia/wikipedia"
WIKI_SIMPLE = "20231101.simple"

# NQ-Swap is built from Natural Questions (open subset): the questions carry
# no passages, and retrieval is deliberately run over an *unrelated* corpus so
# the gold answer can never be surfaced. We reuse the already-cached
# 2WikiMultihopQA validation corpus as that noise pool (zero extra download).
NQ_OPEN_DATASET = "google-research-datasets/nq_open"
NQSWAP_NOISE_CORPUS = ("2wikimultihopqa", "validation")

SUPPORTED_DATASETS = [
    "hotpotqa", "2wikimultihopqa", "musique",
    "popqa", "nqswap", "triviaqa", "wikisimple",
]


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


def _dedup(items: Iterable[str]) -> list[str]:
    """Dedupe a list of gold answers, keeping first occurrence in order.

    Dedup is by normalized-lowercase form (different spelling variants of the
    same answer are common across PopQA/TriviaQA aliases).
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _split_paragraphs(example_id: str, title: str, text: str) -> Iterable[Document]:
    """Split a long Wikipedia-style article into paragraph-level Documents.

    The existing hotpotqa index is paragraph-level, so corpus articles (Simple
    English Wikipedia for PopQA, TriviaQA entity_pages) are split the same way:
    blank-line separated, short junk dropped, each paragraph capped to bound
    BM25 memory and the LLM context fed to the generator.
    """
    title = html.unescape(title).strip()
    for paragraph_id, raw in enumerate(text.split("\n\n")):
        para = html.unescape(raw).strip()
        if len(para) < 80:
            continue
        if len(para) > 1000:
            para = para[:1000].rsplit(" ", 1)[0]
        yield Document(
            id=f"{example_id}:{paragraph_id}",
            title=title,
            text=para,
            example_id=example_id,
            paragraph_id=paragraph_id,
        )


def gold_titles(example, dataset: str = "hotpotqa") -> set[str]:
    """Set of normalized gold supporting-paragraph titles for one example."""
    if dataset in {"popqa", "nqswap", "triviaqa"}:
        # These datasets carry no supporting-fact titles. Consequence for the
        # adaptive env: with an empty gold set, _gold_is_retrievable() returns
        # True, so REJECT is always penalized and the retrieval bonus always
        # fires. That is the right inductive bias here — the answer *is*
        # retrievable (PopQA/TriviaQA) and for NQ-Swap the noise test measures
        # exactly whether the model answers from irrelevant context.
        return set()
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
    elif dataset == "wikisimple":
        # Simple English Wikipedia, used as the retrieval corpus for PopQA
        # (PopQA ships no passages of its own).
        split_str = f"train[:{limit}]" if limit else "train"
        ds = load_dataset(WIKI_SIMPLE_REPO, WIKI_SIMPLE, split=split_str, cache_dir=HF_CACHE_DIR)
    elif dataset == "popqa":
        # Open-domain long-tail QA. Only a test split exists; no passages.
        split_str = f"test[:{limit}]" if limit else "test"
        ds = load_dataset("akariasai/PopQA", split=split_str, cache_dir=HF_CACHE_DIR)
    elif dataset == "nqswap":
        # NQ-Swap: NQ open questions + answers, no passages. Retrieval runs
        # over the unrelated noise corpus (NQSWAP_NOISE_CORPUS) so the gold
        # answer can never be retrieved — the "swap" deliberately removes the
        # evidence, testing hallucination under retrieval noise.
        # NQ-Open has no "test" split on the Hub — only train/validation —
        # so the swap uses the validation split as the eval set.
        split_str = f"validation[:{limit}]" if limit else "validation"
        raw = load_dataset(NQ_OPEN_DATASET, split=split_str, cache_dir=HF_CACHE_DIR)
        return [
            {
                "id": f"nq:{i}",
                "question": ex["question"],
                "answer": (ex.get("answer") or [""])[0],
                "answers": list(ex.get("answer") or []),
            }
            for i, ex in enumerate(raw)
        ]
    elif dataset == "triviaqa":
        # Test split has no public answers, so evaluation uses validation.
        # `rc` config ships entity_pages (Wikipedia evidence) per question.
        split_str = f"validation[:{limit}]" if limit else "validation"
        ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=split_str, cache_dir=HF_CACHE_DIR)
    else:
        split_str = f"{split}[:{limit}]" if limit else split
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_str, cache_dir=HF_CACHE_DIR)

    return ds


def iter_context_paragraphs(examples, dataset: str = "hotpotqa") -> Iterable[Document]:
    """Yield paragraph-level documents from dataset contexts."""
    if dataset in {"popqa", "nqswap"}:
        # No built-in passages; these retrieve over an external corpus
        # (see corpus_spec). Yield nothing.
        return
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
        elif dataset == "wikisimple":
            # Whole articles -> paragraph-level documents (the PopQA corpus).
            example_id = str(example.get("id") or example_index)
            yield from _split_paragraphs(
                example_id, example.get("title", ""), example.get("text", "")
            )
        elif dataset == "triviaqa":
            # Evidence = each question's entity_pages (Wikipedia pages). In the
            # `rc` config entity_pages is a dict of parallel lists, e.g.
            # {"title": [...], "filename": [...], "wiki_context": [...]} — not a
            # list of dicts — so zip the title/wiki_context lists directly.
            example_id = str(example.get("question_id") or example_index)
            pages = example.get("entity_pages") or {}
            titles = pages.get("title") or []
            contexts = pages.get("wiki_context") or []
            for title, wiki_context in zip(titles, contexts):
                yield from _split_paragraphs(example_id, title, wiki_context)
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


def example_answers(example, dataset: str = "hotpotqa") -> list[str]:
    """Gold answers for one example as a list (datasets may accept several).

    HotpotQA-style datasets have a single ``answer``; PopQA/NQ/TriviaQA carry
    answer lists / aliases, scored as max-over-golds.
    """
    if dataset == "popqa":
        out: list[str] = []
        for key in ("s_answers", "e_answers"):
            values = example.get(key) or []
            if not isinstance(values, list):
                values = [values]
            out.extend(str(v) for v in values)
        # Fallbacks in case the actual schema differs from the canonical one.
        for key in ("object", "answer", "possible_answers"):
            value = example.get(key)
            if not value:
                continue
            out.extend(value if isinstance(value, list) else [value])
        return _dedup(out)
    if dataset == "nqswap":
        return _dedup(example.get("answers") or [])
    if dataset == "triviaqa":
        ans = example.get("answer") or {}
        out = [ans.get("value")] if ans.get("value") else []
        out.extend(ans.get("aliases") or [])
        return _dedup(out)
    # Default: HotpotQA / 2WikiMultihopQA / MuSiQue single gold answer.
    return [str(example["answer"])]


def corpus_spec(dataset: str, split: str) -> tuple[str, str]:
    """Which corpus (dataset, split) retrieval indexes for a QA dataset.

    PopQA has no passages and retrieves over the shared Simple English
    Wikipedia corpus; NQ-Swap retrieves over the unrelated 2WikiMultihopQA
    validation corpus so the answer can never be surfaced. Everything else
    retrieves over its own contexts.
    """
    if dataset == "popqa":
        return "wikisimple", "train"
    if dataset == "nqswap":
        return NQSWAP_NOISE_CORPUS
    return dataset, split


def default_index_path(dataset: str, split: str) -> Path:
    """BM25 index path a dataset's retrieval should use (via corpus_spec)."""
    corpus_ds, corpus_split = corpus_spec(dataset, split)
    return ARTIFACTS_DIR / "indexes" / f"bm25_index_{corpus_ds}_{corpus_split}.pkl"
