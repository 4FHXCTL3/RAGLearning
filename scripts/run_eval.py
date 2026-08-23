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

from rag_learning.data import (
    SUPPORTED_DATASETS,
    corpus_spec,
    default_index_path,
    example_answers,
    gold_titles,
    norm_title,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def policy_answer(env, policy, question, gold_answers, golds, max_steps=6):
    """Run one policy-driven episode (framework module 2 inference).

    The policy picks stop/retrieve/reject from the observed state until the
    episode terminates. Returns (answer, contexts). Gold is only used by the
    env for reward/honesty bookkeeping, not to choose the answer.
    """
    import numpy as np

    state = env.reset(question, gold_answers, golds)
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
    parser = argparse.ArgumentParser(description="Run a small QA RAG evaluation.")
    parser.add_argument("--dataset", default="hotpotqa", choices=SUPPORTED_DATASETS,
                        help="Dataset to evaluate.")
    parser.add_argument("--split", default="validation", help="Dataset split, e.g. train or validation.")
    parser.add_argument("--model", default=None,
                        help="Ollama model name to generate with. Defaults to "
                             "LLM_MODEL from .env (e.g. --model llama3:8b).")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to evaluate.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of retrieved paragraphs per question.")
    parser.add_argument("--index", type=Path, default=None,
                        help="Path to a saved BM25 index. Auto-resolved from the "
                             "dataset's retrieval corpus if not set (e.g. PopQA -> "
                             "the Simple English Wikipedia index).")
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
    parser.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["bm25", "dense", "hybrid"],
        help="Index type to use. The provided --index must match this type "
             "(a hybrid index also works for dense and bm25).",
    )
    return parser.parse_args()


def main() -> None:
    from rag_learning.data import load_examples
    from rag_learning.evaluate import compute_em_f1, context_recall
    from rag_learning.index import BM25Index, DenseIndex, HybridIndex, load_any
    from rag_learning.llm import ChatLLM
    from rag_learning.rag import RAGPipeline

    args = parse_args()
    corpus_ds, corpus_split = corpus_spec(args.dataset, args.split)
    if args.index is None:
        args.index = default_index_path(args.dataset, args.split)
    if not args.index.exists():
        raise FileNotFoundError(
            f"Index not found: {args.index}. Build it first with "
            f"`python scripts/build_index.py --dataset {corpus_ds} "
            f"--split {corpus_split} --limit 200 --retrieval-mode {args.retrieval_mode}`."
        )

    examples = load_examples(split=args.split, limit=args.limit, dataset=args.dataset)
    # Auto-detect index type from the pickle so callers can pass either a
    # legacy BM25-only .pkl, a dense-only .pkl, or a combined hybrid .pkl.
    index = load_any(args.index)
    if isinstance(index, HybridIndex) and args.retrieval_mode in {"bm25", "dense", "hybrid"}:
        pass  # HybridIndex supports all three modes
    elif isinstance(index, DenseIndex) and args.retrieval_mode in {"dense", "bm25"}:
        pass
    elif isinstance(index, BM25Index) and args.retrieval_mode == "bm25":
        pass

    # Index coverage check. The standard "smoke test then scale up" workflow
    # is: build_index with a generous --limit, then train on a small subset
    # and eval on a larger subset. If the eval --limit exceeds the index's
    # question coverage, retrieval will silently degrade on the uncovered
    # questions. Heuristic: most datasets have ~5-15 paragraphs per question;
    # if the index has way fewer docs than the eval set needs, warn loudly.
    n_index_docs = len(index.documents)
    n_eval_q = len(examples)
    # Use a conservative lower bound: 3 paragraphs/question minimum. Below
    # this, recall will visibly suffer.
    min_expected_docs = n_eval_q * 3
    if n_index_docs < min_expected_docs:
        print(
            f"\n  WARNING: index may be too small for this eval set.\n"
            f"    index has {n_index_docs} docs, eval set is {n_eval_q} questions\n"
            f"    (rule of thumb: at least {min_expected_docs} docs recommended).\n"
            f"    Retrieval recall on some eval questions will silently degrade.\n"
            f"    -> Re-run `scripts/build_index.py --dataset {args.dataset} "
            f"--split {args.split} --limit {n_eval_q} --retrieval-mode {args.retrieval_mode}` "
            f"to rebuild with full coverage.\n"
        )
    else:
        raise TypeError(
            f"--retrieval-mode {args.retrieval_mode!r} is incompatible with "
            f"the loaded index type {type(index).__name__}."
        )
    pipeline = RAGPipeline(index=index, llm=ChatLLM(model=args.model))

    # Optional: adaptive policy-driven inference (framework module 2).
    policy = None
    env = None
    if args.policy is not None:
        from rag_learning.environment import RAGEnv
        from rag_learning.policy import PolicyNetwork
        from rag_learning.state import StateEncoder, StateNormalizer

        # Cross-LLM transfer: load the StateNormalizer that was fitted during
        # training so the raw state at eval time is normalized with the same
        # mean/var the policy was trained on. Without this, switching the LLM
        # (or even just running with a slightly different LLM temperature)
        # would shift Hsem and the policy would mis-classify.
        state_normalizer = StateNormalizer()
        ckpt_sidecar = args.policy.with_suffix(".ckpt.json")
        ckpt_data: dict = {}
        if ckpt_sidecar.exists():
            try:
                with ckpt_sidecar.open("r", encoding="utf-8") as f:
                    ckpt_data = json.load(f)
                if ckpt_data.get("state_normalizer"):
                    state_normalizer.load(ckpt_data["state_normalizer"])
                    print(
                        f"Loaded state normalizer from checkpoint: "
                        f"mean={[round(float(m), 3) for m in state_normalizer.mean.tolist()]}, "
                        f"std={[round(float(s), 3) for s in (state_normalizer.var ** 0.5).tolist()]}"
                    )
                # If the eval LLM is not in the training LLM list, mention it
                # (this is the supported cross-LLM transfer case).
                train_llms = ckpt_data.get("llm_names") or []
                if train_llms and pipeline.llm.model not in train_llms:
                    print(
                        f"  Note: eval LLM '{pipeline.llm.model}' was not in the "
                        f"training LLMs ({train_llms}). Cross-LLM transfer is "
                        f"supported via the state normalizer."
                    )
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  Warning: failed to read checkpoint sidecar: {exc}")

        policy = PolicyNetwork.load(args.policy)
        env = RAGEnv(
            index=index,
            llm=pipeline.llm,
            encoder=StateEncoder(normalizer=state_normalizer),
            max_retrievals=args.max_retrievals,
            retrieval_mode=args.retrieval_mode,
        )
        print(f"Policy inference enabled: {args.policy}")

    # Train/eval leakage check: if a trained policy is loaded and the training
    # run used the same --split, the evaluation set is not held out. Warn
    # loudly so the user notices in the eval log.
    if args.policy is not None and args.policy.with_suffix(".ckpt.json").exists():
        try:
            with args.policy.with_suffix(".ckpt.json").open("r", encoding="utf-8") as f:
                ckpt = json.load(f)
            trained_split = ckpt.get("progress", {}).get("split")  # may be absent
            if trained_split is None:
                # Older checkpoint — fall back to reading the log
                log_path = (
                    ARTIFACTS_DIR / "train_logs" /
                    f"{args.dataset}_{Path(args.policy).stem.replace('_policy', '')}_train_log.json"
                )
                if log_path.exists():
                    with log_path.open("r", encoding="utf-8") as f:
                        log_cfg = json.load(f).get("config", {})
                    trained_split = log_cfg.get("split")
            if trained_split == args.split:
                print(
                    f"\n  WARNING: train/eval split match: policy was trained on "
                    f"split={trained_split!r} and you are evaluating on the same split. "
                    f"Results reflect in-distribution accuracy, not held-out generalization. "
                    f"Use --split train (or validation) for training and the other for evaluation.\n"
                )
        except (OSError, json.JSONDecodeError):
            pass

    # Name the results file after dataset + model so different dataset/model
    # runs don't overwrite each other (e.g. eval_results_triviaqa_llama3-8b.json).
    if args.results_out is None:
        from rag_learning.config import model_slug
        suffix = "_policy" if args.policy is not None else ""
        args.results_out = ARTIFACTS_DIR / "eval_results" / \
            f"eval_results_{args.dataset}_{model_slug(pipeline.llm.model)}{suffix}.json"

    em_scores: list[float] = []
    f1_scores: list[float] = []
    recall_scores: list[float] = []
    records: list[dict] = []

    for offset, example in enumerate(tqdm(examples, desc="Evaluating"), start=1):
        question = example["question"]
        gold_list = example_answers(example, dataset=args.dataset)
        golds = gold_titles(example, dataset=args.dataset)

        if policy is not None:
            answer, ctx_docs = policy_answer(env, policy, question, gold_list, golds)
            # env contexts are Document objects (no retrieval score).
            ctx_view = [(getattr(d, "title", ""), getattr(d, "id", ""), None) for d in ctx_docs]
            retrieved_text = " ".join((getattr(d, "text", "") or "") for d in ctx_docs)
        else:
            result = pipeline.answer(question, top_k=args.top_k)
            answer = result.answer
            ctx_view = [(c.title, c.id, c.score) for c in result.contexts]
            retrieved_text = " ".join(c.text or "" for c in result.contexts)

        em, f1 = compute_em_f1(answer, gold_list)
        em_scores.append(em)
        f1_scores.append(f1)

        # Step 0 diagnostics: did retrieval actually surface the evidence
        # needed to answer? Title-based for multi-hop datasets (those with
        # `supporting_facts`); answer-in-context for open-domain ones
        # (TriviaQA / PopQA / NQ-Swap) where gold titles don't exist.
        retrieved_titles = {norm_title(t) for t, _id, _s in ctx_view}
        recall, full_hit, evidence_kind = context_recall(
            gold_titles=golds,
            gold_answers=gold_list,
            retrieved_titles=retrieved_titles,
            retrieved_text=retrieved_text,
        )
        if evidence_kind == "answer":
            norm_text = retrieved_text.lower()
            num_gold_answers_found = sum(
                1 for a in gold_list
                if (a or "").strip() and (a or "").strip().lower() in norm_text
            )
            num_gold_retrieved = num_gold_answers_found
        else:
            num_gold_answers_found = 0
            num_gold_retrieved = len(golds & retrieved_titles)
        recall_scores.append(recall)

        category = "correct" if em > 0 else (
            "rejected_unknown" if answer.lower() == "unknown" else "incorrect_answer"
        )
        records.append({
            "example_id": example.get("_id") or example.get("id", f"{offset - 1}"),
            "question": question,
            "gold": gold_list[0] if gold_list else "",
            "golds": gold_list,
            "predict": answer,
            "em": em,
            "f1": f1,
            "evidence_kind": evidence_kind,
            "gold_titles": sorted(golds),
            "num_gold": len(golds),
            "num_gold_retrieved": num_gold_retrieved,
            "gold_answers": gold_list,
            "num_gold_answers": len(gold_list),
            "num_gold_answers_found": num_gold_answers_found,
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
        print(f"Gold:     {gold_list[0] if gold_list else ''}")
        print(f"Predict:  {answer}")
        print(f"EM/F1:    {em:.3f}/{f1:.3f}")
        if evidence_kind == "title":
            print(f"Recall: {recall:.3f} ({num_gold_retrieved}/{len(golds)} gold titles, full_hit={full_hit})")
        else:
            tag = "found" if num_gold_answers_found else "NOT found"
            print(f"Recall: {recall:.3f} (answer {tag} in context, "
                  f"{num_gold_answers_found}/{len(gold_list)} aliases matched, full_hit={full_hit})")
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
    evidence_kind = records[0]["evidence_kind"] if records else "n/a"
    print(f"Evidence kind:     {evidence_kind}")
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
            "dataset": args.dataset,
            "split": args.split,
            "limit": args.limit,
            "top_k": args.top_k,
            "model": pipeline.llm.model,
        },
        "summary": {
            "num_examples": len(records),
            "evidence_kind": records[0]["evidence_kind"] if records else "title",
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