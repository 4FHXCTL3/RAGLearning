from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import DEFAULT_INDEX_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BM25 index from HotpotQA contexts.")
    parser.add_argument("--split", default="validation", help="HotpotQA split, e.g. train or validation.")
    parser.add_argument("--limit", type=int, default=200, help="Number of examples to index for quick experiments.")
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH, help="Output index path.")
    return parser.parse_args()


def main() -> None:
    from rag_learning.data import iter_context_paragraphs, load_examples
    from rag_learning.index import BM25Index

    args = parse_args()
    examples = load_examples(split=args.split, limit=args.limit)
    documents = list(iter_context_paragraphs(examples))
    index = BM25Index(documents)
    index.save(args.output)
    print(f"Loaded {len(examples)} HotpotQA examples.")
    print(f"Indexed {len(documents)} context paragraphs.")
    print(f"Saved BM25 index to {args.output}.")


if __name__ == "__main__":
    main()
