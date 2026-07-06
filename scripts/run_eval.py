from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

# On Windows the default console codec (gbk) crashes when printing non-latin
# characters that appear in Wikipedia titles. Force UTF-8 so evaluation never
# dies on a print statement.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import DEFAULT_INDEX_PATH

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "artifacts" / "eval_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small HotpotQA RAG evaluation.")
    parser.add_argument("--split", default="validation", help="HotpotQA split, e.g. train or validation.")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to evaluate.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved paragraphs per question.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH, help="Path to a saved BM25 index.")
    parser.add_argument(
        "--results-out",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Where to save detailed evaluation results as JSON (used by visualize_results.py).",
    )
    return parser.parse_args()


def main() -> None:
    from rag_learning.data import load_examples
    from rag_learning.evaluate import exact_match_score, f1_score
    from rag_learning.index import BM25Index
    from rag_learning.llm import ChatLLM
    from rag_learning.rag import RAGPipeline

    args = parse_args()
    if not args.index.exists():
        raise FileNotFoundError(
            f"Index not found: {args.index}. Run `python scripts/build_index.py --split {args.split} --limit 200` first."
        )

    examples = load_examples(split=args.split, limit=args.limit)
    index = BM25Index.load(args.index)
    pipeline = RAGPipeline(index=index, llm=ChatLLM())

    em_scores: list[float] = []
    f1_scores: list[float] = []
    records: list[dict] = []

    for offset, example in enumerate(tqdm(examples, desc="Evaluating"), start=1):
        question = example["question"]
        gold_answer = example["answer"]
        result = pipeline.answer(question, top_k=args.top_k)
        answer = result.answer

        em = exact_match_score(answer, gold_answer)
        f1 = f1_score(answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        category = "correct" if em > 0 else (
            "rejected_unknown" if answer.lower() == "unknown" else "incorrect_answer"
        )
        records.append({
            "example_id": example.get("id", f"{offset - 1}"),
            "question": question,
            "gold": gold_answer,
            "predict": answer,
            "em": em,
            "f1": f1,
            "category": category,
            "retrieved_contexts": [
                {"title": ctx.title, "score": ctx.score, "id": ctx.id}
                for ctx in result.contexts
            ],
        })

        print("=" * 80)
        print(f"Example {offset}")
        print(f"Question: {question}")
        print(f"Gold:     {gold_answer}")
        print(f"Predict:  {answer}")
        print(f"EM/F1:    {em:.3f}/{f1:.3f}")
        print("Retrieved:")
        for ctx_index, ctx in enumerate(result.contexts, start=1):
            print(f"  {ctx_index}. {ctx.title} | score={ctx.score:.3f} | id={ctx.id}")

    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print("=" * 80)
    print(f"Average EM: {avg_em:.3f}")
    print(f"Average F1: {avg_f1:.3f}")

    # Persist results so visualize_results.py can reflect this exact run.
    payload = {
        "config": {
            "split": args.split,
            "limit": args.limit,
            "top_k": args.top_k,
            "model": pipeline.llm.model,
        },
        "summary": {
            "num_examples": len(records),
            "avg_em": avg_em,
            "avg_f1": avg_f1,
        },
        "results": records,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    with args.results_out.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"Saved detailed results to {args.results_out}")


if __name__ == "__main__":
    main()