#!/usr/bin/env python
"""Wo-RAG vs Single RAG baseline comparison on multi-hop QA datasets.

Compares two methods on the same set of examples:
  1. Wo-RAG (Without RAG): LLM answers directly from its own knowledge, no retrieval.
  2. Single RAG: Single-round BM25 retrieval + context-grounded generation.

Outputs per-example results and aggregate EM/F1 statistics for both methods,
saved as JSON and Markdown.

Usage:
    # HotpotQA (200 examples)
    python scripts/run_baseline_comparison.py --dataset hotpotqa --split validation --limit 200 --top-k 10

    # 2WikiMultihopQA (200 examples)
    python scripts/run_baseline_comparison.py --dataset 2wikimultihopqa --split validation --limit 200 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import DEFAULT_INDEX_PATH, model_slug
from rag_learning.data import gold_titles

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def build_index_path(dataset: str, split: str = "validation") -> Path:
    """Return the BM25 index path for a given dataset and split."""
    return ARTIFACTS_DIR / "indexes" / f"bm25_index_{dataset}_{split}.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wo-RAG vs Single RAG baseline comparison."
    )
    parser.add_argument("--dataset", default="hotpotqa",
                        help="Dataset: hotpotqa or 2wikimultihopqa.")
    parser.add_argument("--split", default="validation",
                        help="Dataset split, e.g. train or validation.")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of examples to evaluate per method.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of retrieved paragraphs per question for Single RAG.")
    parser.add_argument("--results-out", type=Path, default=None,
                        help="Where to save detailed comparison results as JSON.")
    parser.add_argument("--stats-out", type=Path, default=None,
                        help="Where to save comparison stats as Markdown.")
    return parser.parse_args()


def main() -> None:
    from rag_learning.data import load_examples
    from rag_learning.evaluate import exact_match_score, f1_score
    from rag_learning.index import BM25Index
    from rag_learning.llm import ChatLLM
    from rag_learning.rag import RAGPipeline, direct_answer

    args = parse_args()
    ds = args.dataset

    # Load examples.
    examples = load_examples(split=args.split, limit=args.limit, dataset=ds)

    # Build / load the index for this dataset.
    index_path = build_index_path(ds, args.split)
    if not index_path.exists():
        print(f"Index not found: {index_path}. Building it now...")
        from rag_learning.data import iter_context_paragraphs
        index = BM25Index()
        docs = list(iter_context_paragraphs(examples, dataset=ds))
        index.add_documents(docs)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.save(index_path)
        print(f"Built and saved index to {index_path}")
    else:
        index = BM25Index.load(index_path)

    llm = ChatLLM()
    pipeline = RAGPipeline(index=index, llm=llm)

    # Output paths.
    slug = model_slug(pipeline.llm.model)
    if args.results_out is None:
        args.results_out = ARTIFACTS_DIR / "eval_results" / f"comparison_{ds}_{slug}.json"
    if args.stats_out is None:
        args.stats_out = ARTIFACTS_DIR / "stats" / f"comparison_{ds}_{slug}.md"

    records: list[dict] = []

    for offset, example in enumerate(tqdm(examples, desc=f"Evaluating ({ds})"), start=1):
        question = example["question"]
        gold_answer = example["answer"]

        # ---- Wo-RAG: no retrieval, direct generation ----
        wo_answer = direct_answer(llm, question)
        wo_em = exact_match_score(wo_answer, gold_answer)
        wo_f1 = f1_score(wo_answer, gold_answer)

        # ---- Single RAG: BM25 retrieval + generation ----
        result = pipeline.answer(question, top_k=args.top_k)
        sr_answer = result.answer
        sr_em = exact_match_score(sr_answer, gold_answer)
        sr_f1 = f1_score(sr_answer, gold_answer)

        records.append({
            "example_id": example.get("_id") or example.get("id", f"{offset - 1}"),
            "question": question,
            "gold": gold_answer,
            "wo_rag_answer": wo_answer,
            "wo_rag_em": wo_em,
            "wo_rag_f1": wo_f1,
            "single_rag_answer": sr_answer,
            "single_rag_em": sr_em,
            "single_rag_f1": sr_f1,
        })

        print("=" * 80)
        print(f"Example {offset}")
        print(f"Q: {question}")
        print(f"Gold:      {gold_answer}")
        print(f"Wo-RAG:    {wo_answer}  | EM={wo_em:.3f}  F1={wo_f1:.3f}")
        print(f"SingleRAG: {sr_answer}  | EM={sr_em:.3f}  F1={sr_f1:.3f}")

    # ---- Aggregate statistics ----
    wo_em_avg = sum(r["wo_rag_em"] for r in records) / len(records)
    wo_f1_avg = sum(r["wo_rag_f1"] for r in records) / len(records)
    sr_em_avg = sum(r["single_rag_em"] for r in records) / len(records)
    sr_f1_avg = sum(r["single_rag_f1"] for r in records) / len(records)

    wo_correct = sum(1 for r in records if r["wo_rag_em"] > 0)
    sr_correct = sum(1 for r in records if r["single_rag_em"] > 0)

    # Per-example deltas (Single RAG vs Wo-RAG)
    em_deltas = [r["single_rag_em"] - r["wo_rag_em"] for r in records]
    f1_deltas = [r["single_rag_f1"] - r["wo_rag_f1"] for r in records]
    sr_wins = sum(1 for d in em_deltas if d > 0)
    wo_wins = sum(1 for d in em_deltas if d < 0)
    ties = sum(1 for d in em_deltas if d == 0)

    total = len(records)

    print("=" * 80)
    print(f"Comparison Summary ({ds}, model={pipeline.llm.model})")
    print(f"  Total examples: {total}")
    print(f"  Top-k: {args.top_k}")
    print()
    print(f"  {'Method':<15} {'Avg EM':>8} {'Avg F1':>8} {'Correct':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Wo-RAG':<15} {wo_em_avg:>8.3f} {wo_f1_avg:>8.3f} {wo_correct:>3}/{total}")
    print(f"  {'Single RAG':<15} {sr_em_avg:>8.3f} {sr_f1_avg:>8.3f} {sr_correct:>3}/{total}")
    print()
    print(f"  Head-to-head (EM):  SR wins={sr_wins}  Wo wins={wo_wins}  ties={ties}")
    print(f"  Avg EM delta (SR - Wo): {sum(em_deltas)/total:>+.3f}")
    print(f"  Avg F1 delta (SR - Wo): {sum(f1_deltas)/total:>+.3f}")

    # ---- Persist results ----
    payload = {
        "config": {
            "dataset": ds,
            "split": args.split,
            "limit": args.limit,
            "top_k": args.top_k,
            "model": pipeline.llm.model,
        },
        "summary": {
            "num_examples": total,
            "wo_rag_avg_em": wo_em_avg,
            "wo_rag_avg_f1": wo_f1_avg,
            "wo_rag_correct": wo_correct,
            "single_rag_avg_em": sr_em_avg,
            "single_rag_avg_f1": sr_f1_avg,
            "single_rag_correct": sr_correct,
            "avg_em_delta_sr_minus_wo": sum(em_deltas) / total,
            "avg_f1_delta_sr_minus_wo": sum(f1_deltas) / total,
            "sr_wins_em": sr_wins,
            "wo_wins_em": wo_wins,
            "ties_em": ties,
        },
        "results": records,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    with args.results_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed results to {args.results_out}")

    # ---- Save stats markdown ----
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_out.open("w", encoding="utf-8") as f:
        f.write(f"# Baseline Comparison: Wo-RAG vs Single RAG\n\n")
        f.write(f"**Dataset:** `{ds}`  \n")
        f.write(f"**Model:** `{pipeline.llm.model}`  \n")
        f.write(f"**Split:** `{args.split}`  \n")
        f.write(f"**Examples:** {total}  \n")
        f.write(f"**Top-k (Single RAG):** {args.top_k}  \n\n")
        f.write("## Summary\n\n")
        f.write("| Method | Avg EM | Avg F1 | Correct |\n")
        f.write("|--------|--------|--------|--------|\n")
        f.write(f"| Wo-RAG | {wo_em_avg:.3f} ({wo_em_avg*100:.1f}%) | {wo_f1_avg:.3f} ({wo_f1_avg*100:.1f}%) | {wo_correct}/{total} |\n")
        f.write(f"| Single RAG | {sr_em_avg:.3f} ({sr_em_avg*100:.1f}%) | {sr_f1_avg:.3f} ({sr_f1_avg*100:.1f}%) | {sr_correct}/{total} |\n\n")
        f.write("## Head-to-Head (EM)\n\n")
        f.write(f"- Single RAG wins: {sr_wins}\n")
        f.write(f"- Wo-RAG wins: {wo_wins}\n")
        f.write(f"- Ties: {ties}\n")
        f.write(f"- Avg EM delta (Single RAG − Wo-RAG): **{sum(em_deltas)/total:+.3f}**\n")
        f.write(f"- Avg F1 delta (Single RAG − Wo-RAG): **{sum(f1_deltas)/total:+.3f}**\n")
    print(f"Saved stats to {args.stats_out}")


if __name__ == "__main__":
    main()
