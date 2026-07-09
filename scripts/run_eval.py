from __future__ import annotations

import argparse
import html
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

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _norm_title(title: str) -> str:
    """Normalize a Wikipedia title for gold/retrieved matching.

    Retrieved titles were html.unescaped when the index was built, so we do the
    same to gold supporting-fact titles and compare case-insensitively.
    """
    return html.unescape(title).strip().lower()


def gold_titles(example) -> set[str]:
    """Set of normalized gold supporting-paragraph titles for one example."""
    sf = example.get("supporting_facts") or {}
    titles = sf.get("title", []) if isinstance(sf, dict) else []
    return {_norm_title(t) for t in titles}


def policy_answer(env, policy, question, gold_answer, golds, max_steps=6):
    """Run one policy-driven episode (framework module 2 inference).

    The policy picks stop/retrieve/reject from the observed state until the
    episode terminates. Returns (answer, contexts). Gold is only used by the
    env for reward/honesty bookkeeping, not to choose the answer.
    """
    import numpy as np

    state = env.reset(question, gold_answer, golds)
    step = None
    for _ in range(max_steps):
        action = int(np.argmax(policy.action_probs(state.to_vector())))
        step = env.step(action)
        state = step.state
        if step.done:
            break
    answer = step.answer if step and step.answer is not None else "unknown"
    return answer, env.contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small HotpotQA RAG evaluation.")
    parser.add_argument("--split", default="validation", help="HotpotQA split, e.g. train or validation.")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to evaluate.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of retrieved paragraphs per question.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH, help="Path to a saved BM25 index.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Path to a trained GRPO policy (.npz). When set, uses adaptive "
        "stop/retrieve/reject inference (framework module 2) instead of single-shot retrieval.",
    )
    parser.add_argument(
        "--max-retrievals",
        type=int,
        default=2,
        help="Retrieval budget per question when --policy is used.",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=None,
        help="Where to save detailed evaluation results as JSON (used by visualize_results.py). "
        "Defaults to artifacts/eval_results_<model>.json so different models don't overwrite each other.",
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

    # Optional: adaptive policy-driven inference (framework module 2).
    policy = None
    env = None
    if args.policy is not None:
        from rag_learning.environment import RAGEnv
        from rag_learning.policy import PolicyNetwork
        from rag_learning.state import StateEncoder
        policy = PolicyNetwork.load(args.policy)
        env = RAGEnv(
            index=index,
            llm=pipeline.llm,
            encoder=StateEncoder(),
            max_retrievals=args.max_retrievals,
        )
        print(f"Policy inference enabled: {args.policy}")

    # Name the results file after the model so different LLM runs don't
    # overwrite each other (e.g. eval_results_llama3.2-3b.json).
    if args.results_out is None:
        from rag_learning.config import model_slug
        suffix = "_policy" if args.policy is not None else ""
        args.results_out = ARTIFACTS_DIR / f"eval_results_{model_slug(pipeline.llm.model)}{suffix}.json"

    em_scores: list[float] = []
    f1_scores: list[float] = []
    recall_scores: list[float] = []
    records: list[dict] = []

    for offset, example in enumerate(tqdm(examples, desc="Evaluating"), start=1):
        question = example["question"]
        gold_answer = example["answer"]
        golds = gold_titles(example)

        if policy is not None:
            answer, ctx_docs = policy_answer(env, policy, question, gold_answer, golds)
            # env contexts are Document objects (no retrieval score).
            ctx_view = [(getattr(d, "title", ""), getattr(d, "id", ""), None) for d in ctx_docs]
        else:
            result = pipeline.answer(question, top_k=args.top_k)
            answer = result.answer
            ctx_view = [(c.title, c.id, c.score) for c in result.contexts]

        em = exact_match_score(answer, gold_answer)
        f1 = f1_score(answer, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        # Step 0 diagnostics: did retrieval actually surface the gold
        # supporting paragraphs? Compare gold titles against retrieved titles.
        retrieved_titles = {_norm_title(t) for t, _id, _s in ctx_view}
        hit = golds & retrieved_titles
        recall = len(hit) / len(golds) if golds else 0.0
        full_hit = bool(golds) and golds.issubset(retrieved_titles)
        recall_scores.append(recall)

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
            "gold_titles": sorted(golds),
            "num_gold": len(golds),
            "num_gold_retrieved": len(hit),
            "recall": recall,
            "full_hit": full_hit,
            "category": category,
            "retrieved_contexts": [
                {"title": t, "score": s, "id": _id}
                for t, _id, s in ctx_view
            ],
        })

        print("=" * 80)
        print(f"Example {offset}")
        print(f"Question: {question}")
        print(f"Gold:     {gold_answer}")
        print(f"Predict:  {answer}")
        print(f"EM/F1:    {em:.3f}/{f1:.3f}")
        print(f"Recall: {recall:.3f} ({len(hit)}/{len(golds)} gold, full_hit={full_hit})")
        print(f"Retrieved ({len(ctx_view)} docs):")
        for ctx_index, (t, _id, s) in enumerate(ctx_view, start=1):
            score_str = f"{s:.3f}" if s is not None else "n/a"
            print(f"  {ctx_index}. {t} | score={score_str} | id={_id}")

    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    num_full_hit = sum(1 for r in records if r["full_hit"])
    full_hit_rate = num_full_hit / len(records) if records else 0.0

    # Cross-tab: when retrieval fully surfaced the gold paragraphs, how good is
    # generation? This separates "fix retrieval" from "fix generation".
    full_hit_records = [r for r in records if r["full_hit"]]
    miss_records = [r for r in records if not r["full_hit"]]

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    print("=" * 80)
    print(f"Average EM:        {avg_em:.3f}")
    print(f"Average F1:        {avg_f1:.3f}")
    print(f"Average Recall@{args.top_k}: {avg_recall:.3f}")
    print(f"Full-hit rate@{args.top_k}: {full_hit_rate:.3f} ({num_full_hit}/{len(records)})")
    print("-" * 80)
    print("Diagnosis (EM/F1 conditioned on retrieval):")
    print(f"  When gold fully retrieved ({len(full_hit_records)} q): "
          f"EM={_avg(full_hit_records, 'em'):.3f}  F1={_avg(full_hit_records, 'f1'):.3f}")
    print(f"  When gold missed/partial ({len(miss_records)} q): "
          f"EM={_avg(miss_records, 'em'):.3f}  F1={_avg(miss_records, 'f1'):.3f}")

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
            "avg_recall": avg_recall,
            "full_hit_rate": full_hit_rate,
            "em_given_full_hit": _avg(full_hit_records, "em"),
            "f1_given_full_hit": _avg(full_hit_records, "f1"),
            "em_given_miss": _avg(miss_records, "em"),
            "f1_given_miss": _avg(miss_records, "f1"),
        },
        "results": records,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    with args.results_out.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"Saved detailed results to {args.results_out}")


if __name__ == "__main__":
    main()