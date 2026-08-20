from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import ARTIFACTS_DIR


def _index_path(dataset: str, split: str) -> Path:
    """Return the BM25 index path for a given dataset and split."""
    return ARTIFACTS_DIR / "indexes" / f"bm25_index_{dataset}_{split}.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BM25 index from multi-hop QA dataset contexts.")
    parser.add_argument("--dataset", default="hotpotqa", help="Dataset: hotpotqa or 2wikimultihopqa.")
    parser.add_argument("--split", default="validation", help="Dataset split, e.g. train or validation.")
    parser.add_argument("--limit", type=int, default=200, help="Number of examples to index for quick experiments.")
    parser.add_argument("--output", type=Path, default=None, help="Output index path. Auto-detected from dataset/split if not set.")
    return parser.parse_args()


def main() -> None:
    from rag_learning.data import iter_context_paragraphs, load_examples
    from rag_learning.index import BM25Index

    args = parse_args()
    if args.output is None:
        args.output = _index_path(args.dataset, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples = load_examples(split=args.split, limit=args.limit, dataset=args.dataset)
    documents = list(iter_context_paragraphs(examples, dataset=args.dataset))
    index = BM25Index(documents)
    index.save(args.output)
    print(f"Loaded {len(examples)} examples from {args.dataset} ({args.split} split).")
    print(f"Indexed {len(documents)} context paragraphs.")
    print(f"Saved BM25 index to {args.output}.")


if __name__ == "__main__":
    main()
