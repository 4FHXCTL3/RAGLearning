from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import ARTIFACTS_DIR


def _index_path(dataset: str, splits: list[str], out: Path | None) -> Path:
    if out is not None:
        return out
    if len(splits) == 1:
        return ARTIFACTS_DIR / "indexes" / f"bm25_index_{dataset}_{splits[0]}.pkl"
    # Multi-split combined: name reflects all splits joined by '+' so the
    # ordering of --split args doesn't change the filename.
    suffix = "+".join(splits)
    return ARTIFACTS_DIR / "indexes" / f"bm25_index_{dataset}_{suffix}.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BM25 / dense / hybrid index from one or more dataset splits."
    )
    parser.add_argument("--dataset", default="hotpotqa", help="Dataset: hotpotqa, triviaqa, ...")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Dataset split(s). Repeat the flag to combine multiple splits into one index "
             "(e.g. --split train --split validation). Default: validation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of examples to index PER SPLIT. Default: all available.",
    )
    parser.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["bm25", "dense", "hybrid"],
        help="Index type to build. 'hybrid' = BM25 + dense + RRF (recommended).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output index path. Auto-detected from dataset/splits if not set.",
    )
    return parser.parse_args()


def _dedupe_documents(documents: list) -> list:
    """Drop exact-duplicate paragraphs (same title + text). BM25 tokenization
    and embedding are both O(N), so deduping saves real time on the
    train+validation combined build where some paragraphs repeat."""
    seen: set[tuple[str, str]] = set()
    out: list = []
    for doc in documents:
        key = (doc.title, doc.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def main() -> None:
    from rag_learning.data import iter_context_paragraphs, load_examples
    from rag_learning.embed import Embedder
    from rag_learning.index import BM25Index, DenseIndex, HybridIndex

    args = parse_args()
    if args.split is None or len(args.split) == 0:
        args.split = ["validation"]
    if args.output is None:
        args.output = _index_path(args.dataset, args.split, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load + dedupe paragraphs across all requested splits.
    documents: list = []
    n_examples_per_split: dict[str, int] = {}
    for split in args.split:
        examples = load_examples(split=split, limit=args.limit, dataset=args.dataset)
        n_examples_per_split[split] = len(examples)
        documents.extend(iter_context_paragraphs(examples, dataset=args.dataset))
    documents = _dedupe_documents(documents)
    n_dups = sum(n_examples_per_split.values())  # before dedup we just had counts; this is informational
    print(
        f"Loaded {n_examples_per_split} examples from {args.dataset} "
        f"(splits: {args.split}); {len(documents)} unique paragraphs after dedup."
    )

    def _embed_batched(texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed in small batches with progress logging.

        Ollama's `/api/embed` endpoint:
          - times out (httpx default 60s) when called with thousands of texts
            in one request
          - returns 400 if any single text exceeds the model's context window
            (nomic-embed-text: 8192 tokens, ~24k chars)
        Batching avoids the first; truncating each text to a safe char budget
        avoids the second.
        """
        # Truncate each text to ~8000 chars to leave headroom for tokenization.
        # Wikipedia paragraphs longer than this are usually boilerplate; the
        # first chunk captures the lead/topic signal we need for retrieval.
        MAX_CHARS = 8000
        truncated = [t if len(t) <= MAX_CHARS else t[:MAX_CHARS] for t in texts]
        n_truncated = sum(1 for o, t in zip(texts, truncated) if len(o) > MAX_CHARS)
        if n_truncated:
            print(f"  Truncated {n_truncated} over-long paragraphs to {MAX_CHARS} chars.")

        embedder = Embedder()
        all_vecs: list[np.ndarray] = []
        n = len(truncated)
        import time as _time
        start = _time.time()
        for i in range(0, n, batch_size):
            batch = truncated[i : i + batch_size]
            try:
                vecs = embedder.embed(batch)  # cache-aware: re-runs are O(1)
            except Exception as exc:
                # Print the full error from Ollama so the user can diagnose
                # (e.g. which text in the batch tripped the model).
                print(f"\n  [embed] batch {i}-{i+len(batch)} failed: {exc!r}")
                if hasattr(exc, "response") and exc.response is not None:
                    try:
                        body = exc.response.text
                        print(f"  [embed] Ollama response body: {body[:500]}")
                    except Exception:
                        pass
                # Find the offending long text in the batch, if any.
                for j, t in enumerate(batch):
                    if len(t) > 20000:
                        print(f"  [embed]   batch[{j}] is {len(t)} chars "
                              f"(first 200): {t[:200]!r}")
                raise
            all_vecs.append(vecs)
            done = min(i + batch_size, n)
            elapsed = _time.time() - start
            rate = done / max(elapsed, 0.1)
            eta = (n - done) / max(rate, 0.1)
            print(f"  embedded {done}/{n} ({100*done/n:.0f}%) "
                  f"in {elapsed:.0f}s — {rate:.0f} docs/s, ETA {eta:.0f}s")
        return np.concatenate(all_vecs, axis=0)

    if args.retrieval_mode == "bm25":
        index = BM25Index(documents)
        index.save(args.output)
    elif args.retrieval_mode == "dense":
        texts = [f"{d.title}. {d.text}" for d in documents]
        print(f"Embedding {len(texts)} paragraphs (batched)...")
        embeddings = _embed_batched(texts)
        index = DenseIndex(documents, embeddings)
        index.save(args.output)
    elif args.retrieval_mode == "hybrid":
        bm25 = BM25Index(documents)
        texts = [f"{d.title}. {d.text}" for d in documents]
        print(f"Embedding {len(texts)} paragraphs (batched)...")
        embeddings = _embed_batched(texts)
        dense = DenseIndex(documents, embeddings)
        index = HybridIndex(bm25, dense)
        index.save(args.output)
    else:
        raise ValueError(f"Unknown retrieval_mode={args.retrieval_mode!r}")

    print(f"Built {type(index).__name__} with {len(index.documents)} documents.")
    print(f"Saved index to {args.output}.")


if __name__ == "__main__":
    main()
