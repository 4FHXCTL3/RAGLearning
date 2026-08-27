from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

import numpy as np

from rag_learning.data import (
    SUPPORTED_DATASETS,
    corpus_spec,
    default_index_path,
    example_answers,
    gold_titles,
    norm_title,
)
from rag_learning.environment import RAGEnv
from rag_learning.evaluate import compute_em_f1, context_recall
from rag_learning.llm import ChatLLM
from rag_learning.rag import RAGPipeline

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def policy_answer(env, policy, question, gold_answers, golds, max_steps=6):
    """Run one policy-driven episode (framework module 2 inference).

    The policy picks stop/retrieve/reject from the observed state until the
    episode terminates. Returns (answer, contexts). Gold is only used by the
    env for reward/honesty bookkeeping, not to choose the answer.
    """
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


def _evaluate_one(
    idx: int,
    example: dict,
    *,
    args,
    pipeline,
    policy,
    encoder_template,
    index,
    shared_embedder=None,
) -> tuple[int, dict]:
    """Process a single example end-to-end. Returns (idx, record).

    All shared, read-mostly resources (index, llm client, embedder, policy,
    state normalizer) are taken from the parent thread; the per-question env
    is constructed here so each worker has its own mutable state. This makes
    ``run_eval`` trivially safe to run in parallel via ThreadPoolExecutor —
    the LLM call is the dominant cost, and local Ollama happily handles
    ``workers`` concurrent requests to the same model.
    """
    question = example["question"]
    gold_list = example_answers(example, dataset=args.dataset)
    golds = gold_titles(example, dataset=args.dataset)

    if policy is not None:
        # Per-worker env so concurrent rollouts don't clobber each other.
        env = RAGEnv(
            index=index,
            llm=pipeline.llm,
            encoder=encoder_template,
            max_retrievals=args.max_retrievals,
            retrieval_mode=args.retrieval_mode,
            embedder=shared_embedder,
        )
        answer, ctx_docs = policy_answer(env, policy, question, gold_list, golds)
        ctx_view = [(getattr(d, "title", ""), getattr(d, "id", ""), None) for d in ctx_docs]
        retrieved_text = " ".join((getattr(d, "text", "") or "") for d in ctx_docs)
    else:
        result = pipeline.answer(question, top_k=args.top_k)
        answer = result.answer
        ctx_view = [(c.title, c.id, c.score) for c in result.contexts]
        retrieved_text = " ".join(c.text or "" for c in result.contexts)

    em, f1 = compute_em_f1(answer, gold_list)

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

    category = "correct" if em > 0 else (
        "rejected_unknown" if answer.lower() == "unknown" else "incorrect_answer"
    )
    record = {
        "example_id": example.get("_id") or example.get("id", str(idx)),
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
    }
    return idx, record


def _print_example(offset: int, record: dict) -> None:
    """Pretty-print a single example's diagnostics. Cheap; called serially
    from the main thread after all examples are processed (or inline in
    --workers 1 mode)."""
    print("=" * 80)
    print(f"Example {offset}")
    print(f"Question: {record['question']}")
    print(f"Gold:     {record['gold']}")
    print(f"Predict:  {record['predict']}")
    print(f"EM/F1:    {record['em']:.3f}/{record['f1']:.3f}")
    if record["evidence_kind"] == "title":
        print(
            f"Recall: {record['recall']:.3f} "
            f"({record['num_gold_retrieved']}/{record['num_gold']} gold titles, "
            f"full_hit={record['full_hit']})"
        )
    else:
        tag = "found" if record["num_gold_answers_found"] else "NOT found"
        print(
            f"Recall: {record['recall']:.3f} (answer {tag} in context, "
            f"{record['num_gold_answers_found']}/{record['num_gold_answers']} "
            f"aliases matched, full_hit={record['full_hit']})"
        )
    print(f"Retrieved ({len(record['retrieved_contexts'])} docs):")
    for ctx_index, ctx in enumerate(record["retrieved_contexts"], start=1):
        score = ctx.get("score")
        score_str = f"{score:.3f}" if score is not None else "n/a"
        print(f"  {ctx_index}. {ctx['title']} | score={score_str} | id={ctx['id']}")


# ---------------------------------------------------------------------------
# Incremental sidecars — extend an existing eval run without re-evaluating
# already-done examples. Same idea as build_index's sidecar system, just
# per-example: one JSONL line per record, append-only during the run.
# ---------------------------------------------------------------------------

def _records_path(json_path: Path) -> Path:
    """JSONL sidecar path for per-example records.

    Example: ``eval_results_foo.json`` → ``eval_results_foo_records.jsonl``.
    """
    return json_path.with_name(json_path.stem + "_records.jsonl")


def _load_existing_records(
    json_path: Path, records_path: Path
) -> tuple[list[dict], set[str]]:
    """Load existing records from the JSONL sidecar (preferred) or the
    legacy summary JSON. Returns ``(records, completed_ids)``.

    The JSONL is the source of truth for incremental eval. If only the
    summary JSON exists (an old run before this feature), we extract the
    records from its ``results`` block; ``__idx`` is re-assigned from the
    list position so the sort-by-original-order works downstream.
    """
    records: list[dict] = []
    if records_path.exists():
        with records_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    elif json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
        for i, rec in enumerate(data.get("results", [])):
            rec = dict(rec)  # copy so we don't mutate the in-memory dict
            rec["__idx"] = i
            records.append(rec)
    completed_ids = {
        rec.get("example_id") for rec in records if rec.get("example_id") is not None
    }
    return records, completed_ids


def _append_record(records_path: Path, record: dict) -> None:
    """Append a single record (with the original example index) to the
    JSONL sidecar. The writer holds an exclusive open so concurrent runs
    on the same output file would race, but normal use is one process."""
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class BufferedJSONLWriter:
    """Append-only JSONL writer with batched flushes.

    For TriviaQA-scale runs (2000 examples) the naive `_append_record`
    does one open/write/close per record = 2000 fsync-flavored open
    syscalls. With ``flush_every=64`` the open happens once, the
    writes are coalesced into a small ``write`` syscall burst every
    64 records, and a final flush on context exit guarantees the
    sidecar is on disk before the script returns. The writer is
    thread-safe (a single ``_lock`` guards the in-memory list + file
    handle), so worker threads can ``.write()`` concurrently.
    """

    def __init__(self, path: Path, flush_every: int = 64):
        self.path = path
        self.flush_every = flush_every
        self._buf: list[str] = []
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._buf.append(line)
            if len(self._buf) >= self.flush_every:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        self._f.write("\n".join(self._buf) + "\n")
        self._f.flush()
        self._buf.clear()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            self._f.close()

    def __enter__(self) -> "BufferedJSONLWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _print_running_summary(
    em_scores: list, f1_scores: list, recall_scores: list, full_hits: int,
    n_done: int, n_total: int,
) -> None:
    """One-line summary of cumulative metrics so far. ``n_done`` includes
    both pre-existing and newly-evaluated examples; ``n_total`` is the
    size of the full examples list (i.e. what a no-skip run would touch)."""
    n = len(em_scores)
    if not n:
        return
    avg_em = sum(em_scores) / n
    avg_f1 = sum(f1_scores) / n
    avg_recall = sum(recall_scores) / n
    full_hit_pct = 100 * full_hits / n
    print(
        f"\n[Progress: {n_done}/{n_total}] "
        f"EM={avg_em:.3f}  F1={avg_f1:.3f}  Recall={avg_recall:.3f}  "
        f"FullHit={full_hit_pct:.1f}% ({full_hits}/{n})",
        flush=True,
    )


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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers to process examples. Each worker "
             "issues its own LLM call, so this is the LLM-call concurrency. "
             "Default 4 matches a typical local Ollama `OLLAMA_NUM_PARALLEL`. "
             "Use --workers 1 to run serially (preserves the old per-example "
             "print order during processing).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print a one-line running summary (EM, F1, Recall, full-hit rate) "
             "every N examples. 0 = silent (just the final summary). 1 = print "
             "the full per-example diagnostics block for every example "
             "(verbose, like the old default). Default 50 strikes a balance: "
             "you see live progress without drowning the terminal.",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help="Extend an existing evaluation by adding more examples instead of "
             "re-evaluating from scratch. Reuses the JSONL records sidecar to "
             "skip examples whose IDs are already done. Useful when you started "
             "with --limit 200 and want to expand to --limit 1000 without "
             "re-paying the LLM cost for the 200 you already evaluated. The "
             "final summary JSON is rebuilt from the JSONL at the end.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate every example even if a results JSON already exists. "
             "Implies --print-every 0 unless overridden. Existing records are "
             "discarded.",
    )
    return parser.parse_args()


def main() -> None:
    from rag_learning.data import load_examples
    from rag_learning.index import BM25Index, DenseIndex, HybridIndex, load_any

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
    else:
        raise TypeError(
            f"--retrieval-mode {args.retrieval_mode!r} is incompatible with "
            f"the loaded index type {type(index).__name__}."
        )

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
    pipeline = RAGPipeline(index=index, llm=ChatLLM(model=args.model))

    # Build ONE shared Embedder (when needed) and hand it to every worker's
    # RAGEnv / StateEncoder. Without this, N workers would each spin up an
    # independent httpx client and an independent embed cache — same questions
    # + same docs would be re-embedded N times. The httpx client is the
    # expensive part (TCP connect, JSON serialization); the cache only saves
    # the *model inference* cost which is the dominant fraction here.
    shared_embedder = None
    if args.retrieval_mode in {"dense", "hybrid"} or args.policy is not None:
        from rag_learning.embed import Embedder
        shared_embedder = Embedder()

    # Optional: adaptive policy-driven inference (framework module 2).
    policy = None
    env = None
    if args.policy is not None:
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
            encoder=StateEncoder(embedder=shared_embedder, normalizer=state_normalizer),
            max_retrievals=args.max_retrievals,
            retrieval_mode=args.retrieval_mode,
            embedder=shared_embedder,
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
    # `results` collects (idx, record) pairs from workers. The serial path
    # appends in original order; the parallel path appends in completion
    # order — we re-sort by idx at the end so the printed record order and
    # the persisted JSON match the input file.
    results: list[tuple[int, dict]] = []

    # Build a template encoder (or None) so workers can construct per-question
    # envs from the same shared resources (index, llm, encoder, normalizer).
    # We do NOT share a single env across workers — each worker's env holds
    # mutable per-episode state (current_k, contexts, _last_state) that must
    # not be clobbered by a concurrent rollout.
    encoder_template = env.encoder if env is not None else None

    # ---- Incremental sidecar: skip examples already evaluated. ----
    records_path = _records_path(args.results_out)
    existing_records: list[dict] = []
    existing_ids: set[str] = set()
    if args.extend and not args.force and args.results_out.exists():
        existing_records, existing_ids = _load_existing_records(
            args.results_out, records_path,
        )
        # If only the legacy JSON existed, populate the JSONL sidecar now
        # so future --extend runs can use the faster path.
        if existing_records and not records_path.exists():
            records_path.parent.mkdir(parents=True, exist_ok=True)
            with records_path.open("w", encoding="utf-8") as f:
                for rec in existing_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elif args.results_out.exists() and not args.force:
        print(f"Results already exist at {args.results_out}.")
        print("  Use --force to re-evaluate from scratch.")
        print("  Use --extend to add more examples without re-evaluating done ones.")
        return
    elif args.force and args.results_out.exists():
        # Drop the sidecar so we start clean.
        if records_path.exists():
            records_path.unlink()

    # Filter the examples list to only those not yet evaluated. Keep the
    # original index so the final JSON's `results` is in the input order.
    if existing_ids:
        todo: list[tuple[int, dict]] = [
            (i, ex) for i, ex in enumerate(examples)
            if (ex.get("_id") or ex.get("id", "")) not in existing_ids
        ]
        print(
            f"  --extend: {len(existing_ids)} examples already evaluated, "
            f"{len(todo)} new to run (out of {len(examples)} total).",
            flush=True,
        )
    else:
        todo = list(enumerate(examples))

    if not todo:
        print("Nothing to evaluate — all examples are already in the sidecar.")

    # Decide the per-example print behaviour from --print-every. Done
    # unconditionally so the post-loop deferred print (parallel mode)
    # can use the same flag.
    print_each = args.print_every == 1
    summary_every = max(args.print_every, 1)
    n_full_hits = 0
    n_done_so_far = len(existing_records)

    if todo:
        # Pre-seed running stats from existing records so the running
        # summary reflects the cumulative picture, not just new evals.
        for rec in existing_records:
            em_scores.append(rec["em"])
            f1_scores.append(rec["f1"])
            recall_scores.append(rec["recall"])
        n_full_hits = sum(1 for r in existing_records if r["full_hit"])

        if args.workers <= 1:
            # Serial path. The BufferedJSONLWriter is used here too — same
            # one writer per process, the batching just amortizes the per-line
            # open/write/close across the whole run rather than only the
            # parallel branch.
            with BufferedJSONLWriter(records_path) as sidecar:
                for orig_idx, example in tqdm(
                    todo, total=len(todo), desc="Evaluating",
                    initial=len(existing_records),  # so the bar's "done" is right
                ):
                    idx, record = _evaluate_one(
                        orig_idx, example,
                        args=args, pipeline=pipeline, policy=policy,
                        encoder_template=encoder_template, index=index,
                        shared_embedder=shared_embedder,
                    )
                    record["__idx"] = orig_idx
                    results.append((idx, record))
                    em_scores.append(record["em"])
                    f1_scores.append(record["f1"])
                    recall_scores.append(record["recall"])
                    n_full_hits += int(record["full_hit"])
                    sidecar.write(record)
                    n_done_so_far += 1
                    if print_each:
                        _print_example(n_done_so_far, record)
                    if args.print_every > 1 and n_done_so_far % summary_every == 0:
                        _print_running_summary(
                            em_scores, f1_scores, recall_scores, n_full_hits,
                            n_done_so_far, len(examples),
                        )
        else:
            # Parallel path: dispatch to a thread pool. The BufferedJSONLWriter
            # is thread-safe so all 8 workers can `.write()` concurrently
            # without serializing at the OS file-handle level.
            pbar = tqdm(
                total=len(examples),
                initial=len(existing_records),
                desc=f"Evaluating (workers={args.workers})",
            )
            with BufferedJSONLWriter(records_path) as sidecar, \
                    ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {
                    ex.submit(
                        _evaluate_one,
                        orig_idx, example,
                        args=args, pipeline=pipeline, policy=policy,
                        encoder_template=encoder_template, index=index,
                        shared_embedder=shared_embedder,
                    ): orig_idx
                    for orig_idx, example in todo
                }
                for fut in as_completed(futures):
                    idx, record = fut.result()
                    record["__idx"] = idx
                    results.append((idx, record))
                    em_scores.append(record["em"])
                    f1_scores.append(record["f1"])
                    recall_scores.append(record["recall"])
                    n_full_hits += int(record["full_hit"])
                    sidecar.write(record)
                    pbar.update(1)
                    n_done_so_far += 1
                    if args.print_every > 1 and n_done_so_far % summary_every == 0:
                        _print_running_summary(
                            em_scores, f1_scores, recall_scores, n_full_hits,
                            n_done_so_far, len(examples),
                        )
            pbar.close()

    # ---- Reconstruct the full records list in original input order. ----
    # The JSONL is the source of truth, but for a 2000-question run we just
    # wrote every new record through the BufferedJSONLWriter above — re-reading
    # the entire file would be O(N) I/O for no information we don't already
    # have in memory. Combine the pre-seeded existing records with the
    # freshly-evaluated ones, dedup by ``example_id`` (NOT ``__idx``: under
    # --extend the new ``todo`` re-numbers from 0 so ``__idx`` collisions
    # between old and new records are expected), and sort by ``__idx`` once.
    combined = list(existing_records)
    new_records_by_id: dict[str, dict] = {}
    for _idx, rec in results:
        eid = rec.get("example_id")
        if eid is not None:
            new_records_by_id[eid] = rec
        else:
            # No example_id (shouldn't happen, but be defensive) — keep by
            # __idx so we don't drop it.
            new_records_by_id[f"__idx:{rec.get('__idx')}"] = rec
    # Replace any old record with the same example_id with the fresh one.
    combined = [
        new_records_by_id.get(rec.get("example_id") or f"__idx:{rec.get('__idx')}", rec)
        for rec in combined
    ]
    # Append any new records that don't exist in the old set yet.
    seen_ids = {rec.get("example_id") for rec in combined}
    for _idx, rec in results:
        eid = rec.get("example_id")
        if eid not in seen_ids:
            combined.append(rec)
            seen_ids.add(eid)
    combined.sort(key=lambda r: r.get("__idx", float("inf")))

    # In serial mode per-example blocks were printed inline; in parallel
    # mode, --print-every 1 defers them to here. We only print records
    # that were evaluated in *this* run (so --extend doesn't re-spam
    # the terminal with already-seen diagnostics). Use ``example_id``
    # to detect "this run" — see comment above on why ``__idx`` is
    # unreliable under --extend.
    if print_each and args.workers > 1 and results:
        new_ids = {rec.get("example_id") for _idx, rec in results}
        for rec in combined:
            if rec.get("example_id") in new_ids:
                _print_example(rec["__idx"] + 1, rec)

    records = combined  # the final, sorted list (used by summary + payload)

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