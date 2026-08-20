#!/usr/bin/env python
"""GRPO policy training for adaptive RAG (CLARITY-RAG Framework Module 2).

Trains a PolicyNetwork on a multi-hop QA dataset using GRPO (Group Relative
Policy Optimization). Supports checkpoint save/resume so training can survive
interruptions (power loss, crash, etc.) and epochs can be extended without
retraining from scratch.

Resume is AUTOMATIC: re-running the same command picks up the latest checkpoint
mid-epoch exactly where it stopped (including the question it was on). Use
--from-scratch to discard a previous run. The training log is rewritten at
every checkpoint, so scripts/visualize_training.py --watch can monitor it live.

Usage:
    # Train from scratch (3 epochs, 100 examples, 2WikiMultihopQA)
    python scripts/train_policy.py --dataset 2wikimultihopqa --limit 100 --epochs 3

    # Re-run the same command after a power loss -> resumes automatically.
    # (A completed run with the same --epochs exits immediately; bump --epochs
    # to extend training.)

    # Discard an old run and start fresh (overwrites the saved policy)
    python scripts/train_policy.py --dataset 2wikimultihopqa --limit 100 --epochs 3 --from-scratch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import model_slug
from rag_learning.data import SUPPORTED_DATASETS, corpus_spec, example_answers, gold_titles
from rag_learning.policy import PolicyNetwork, RunningBaseline

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_index_path(dataset: str, split: str = "validation") -> Path:
    """Return the BM25 index path for a given dataset and split.

    Split is encoded in the filename so train and validation indexes are
    distinct — avoids silently reusing a small validation index for training.
    """
    return ARTIFACTS_DIR / "indexes" / f"bm25_index_{dataset}_{split}.pkl"


def checkpoint_path(policy_path: Path) -> Path:
    """Return the sidecar checkpoint JSON path for a given .npz policy path."""
    return policy_path.with_suffix(".ckpt.json")


def _build_index(dataset: str, split: str, limit: int | None) -> Path:
    """Build and save a BM25 index for the given dataset, return its path."""
    from rag_learning.data import iter_context_paragraphs, load_examples
    from rag_learning.index import BM25Index

    idx_path = build_index_path(dataset, split)
    if idx_path.exists():
        print(f"Index already exists at {idx_path}, loading...")
        return idx_path

    print(f"Building BM25 index for {dataset} ({split}, limit={limit})...")
    examples = load_examples(split=split, limit=limit, dataset=dataset)
    documents = list(iter_context_paragraphs(examples, dataset=dataset))
    index = BM25Index(documents)
    index.save(idx_path)
    print(f"Indexed {len(documents)} paragraphs from {len(examples)} examples.")
    return idx_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO policy training for adaptive RAG."
    )
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=SUPPORTED_DATASETS,
                        help="Dataset to train on.")
    parser.add_argument("--split", default="train",
                        help="Dataset split (train / validation / test).")
    parser.add_argument("--model", default=None,
                        help="Ollama model name to generate with. Defaults to "
                             "LLM_MODEL from .env (e.g. --model llama3:8b).")
    parser.add_argument("--offline", action="store_true",
                        help="Force HuggingFace datasets offline mode "
                             "(use local cache only; no downloads).")
    parser.add_argument("--index-limit", type=int, default=None,
                        help="Number of corpus examples to index when the "
                             "retrieval index is auto-built (e.g. Simple English "
                             "Wikipedia behind PopQA). Defaults to --limit.")
    parser.add_argument("--limit", type=int, default=200,
                        help="Number of training examples.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Total training epochs.")
    parser.add_argument("--group-size", type=int, default=3,
                        help="GRPO group size G (number of sampled actions per state).")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="GRPO learning rate.")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="GRPO clip epsilon.")
    parser.add_argument("--max-retrievals", type=int, default=2,
                        help="Maximum number of retrieve actions per episode.")
    parser.add_argument("--initial-k", type=int, default=5,
                        help="Number of documents retrieved on first step.")
    parser.add_argument("--k-increment", type=int, default=5,
                        help="Additional documents per retrieve action.")
    parser.add_argument("--sample-temperature", type=float, default=0.7,
                        help="Temperature for answer sampling (0 = greedy).")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N questions within an epoch.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--index", type=Path, default=None,
                        help="Path to BM25 index. Auto-built if not found.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Policy output path (.npz). Auto-generated if not set.")
    parser.add_argument("--log-out", type=Path, default=None,
                        help="Training log output path (.json). Auto-generated if not set.")
    parser.add_argument("--resume", action="store_true",
                        help="Deprecated: auto-resume from the latest checkpoint "
                             "is now the default. Kept for backward compatibility.")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Ignore any existing checkpoint/policy and start "
                             "training from scratch (overwrites the saved policy).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from rag_learning.data import load_examples
    from rag_learning.embed import Embedder
    from rag_learning.environment import RAGEnv, RewardWeights
    from rag_learning.index import BM25Index
    from rag_learning.llm import ChatLLM
    from rag_learning.state import StateEncoder

    args = parse_args()
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    rng = np.random.default_rng(args.seed)

    # --- resolve paths ---
    llm = ChatLLM(model=args.model)
    slug = model_slug(llm.model)

    if args.output is None:
        args.output = ARTIFACTS_DIR / "policies" / f"{args.dataset}_{slug}_policy.npz"
    if args.log_out is None:
        args.log_out = ARTIFACTS_DIR / "train_logs" / f"{args.dataset}_{slug}_train_log.json"

    ckpt_path = checkpoint_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_out.parent.mkdir(parents=True, exist_ok=True)

    # --- checkpoint state (used for auto-resume) ---
    # `saved_epoch_order` lets us replay the exact shuffled order of the epoch
    # that was interrupted, so resume continues mid-epoch instead of skipping
    # or re-doing questions.
    start_epoch = 0
    start_q_idx = 0
    epoch_returns: list[float] = []
    epoch_objectives: list[float] = []
    action_counts: dict[str, int] = {"stop": 0, "retrieve": 0, "reject": 0}
    action_count_history: list[dict] = []
    steps_log: list[dict] = []
    step_counter = 0
    prior_elapsed = 0.0
    saved_epoch_order: list[int] | None = None
    completed_epochs = 0

    # --- load or init policy (auto-resume unless --from-scratch) ---
    policy: PolicyNetwork
    if not args.from_scratch and args.output.exists():
        policy = PolicyNetwork.load(args.output)
        if ckpt_path.exists():
            with ckpt_path.open("r", encoding="utf-8") as f:
                ckpt = json.load(f)
            progress = ckpt.get("progress", {})
            start_epoch = progress.get("current_epoch", 0)
            start_q_idx = progress.get("current_question_idx", 0)
            completed_epochs = progress.get("completed_epochs", 0)
            prior_elapsed = progress.get("elapsed_s", 0.0)
            epoch_returns = ckpt.get("metrics", {}).get("epoch_returns", [])
            epoch_objectives = ckpt.get("metrics", {}).get("epoch_objectives", [])
            action_counts = ckpt.get("metrics", {}).get("action_counts",
                                                         {"stop": 0, "retrieve": 0, "reject": 0})
            action_count_history = ckpt.get("action_count_history", [])
            saved_epoch_order = ckpt.get("epoch_order")
            steps_log = ckpt.get("steps_log") or []
            step_counter = len(steps_log)  # each step appends exactly one entry
            print(f"Resumed from checkpoint: epoch={start_epoch}/{args.epochs}, "
                  f"question_idx={start_q_idx}, completed_epochs={completed_epochs}, "
                  f"steps_so_far={step_counter}")
        else:
            print(f"Warning: found policy at {args.output} but no checkpoint at "
                  f"{ckpt_path}. Starting from scratch.")
    else:
        policy = PolicyNetwork(seed=args.seed)
        print("Initialized new policy network.")

    # Check if already fully trained.
    if completed_epochs >= args.epochs:
        print(f"Training already completed: {completed_epochs} >= {args.epochs} epochs. "
              f"Use --epochs {completed_epochs + 1} (or higher) to continue training.")
        return

    # --- build / load index (retrieve over the dataset's corpus) ---
    # popqa -> Simple English Wikipedia, nqswap -> 2WikiMultihopQA noise corpus,
    # everything else -> its own contexts (see corpus_spec).
    corpus_ds, corpus_split = corpus_spec(args.dataset, args.split)
    index_limit = args.index_limit or args.limit
    if args.index is not None and args.index.exists():
        index = BM25Index.load(args.index)
    else:
        idx_path = _build_index(corpus_ds, corpus_split, index_limit)
        index = BM25Index.load(idx_path)
    print(f"Loaded BM25 index with {len(index.documents)} documents.")

    # --- load training examples ---
    examples = load_examples(split=args.split, limit=args.limit, dataset=args.dataset)
    total_questions = len(examples)
    if total_questions == 0:
        print("No training examples loaded. Exiting.")
        return
    print(f"Loaded {total_questions} training examples from {args.dataset} ({args.split}).")

    # --- init modules ---
    embedder = Embedder()
    encoder = StateEncoder(embedder=embedder)
    weights = RewardWeights()
    env = RAGEnv(
        index=index,
        llm=llm,
        encoder=encoder,
        weights=weights,
        initial_k=args.initial_k,
        k_increment=args.k_increment,
        max_retrievals=args.max_retrievals,
        group_size=args.group_size,
        sample_temperature=args.sample_temperature,
    )

    # --- training loop ---
    print(f"\nStarting training: {args.epochs} epoch(s), {total_questions} questions, "
          f"group_size={args.group_size}, lr={args.lr}")
    print(f"Output: {args.output}")
    print(f"Log:    {args.log_out}")
    print()

    # Running baseline for stable advantage computation (Fix 1).
    baseline = RunningBaseline(alpha=0.9)

    # `prior_elapsed` folds in wall-time from previous (interrupted) sessions so
    # the log's total_time_s reflects cumulative training time across resumes.
    total_start_time = time.time() - prior_elapsed
    ema_return = None  # running EMA of group return shown in the progress bar

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_return_sum = 0.0
        epoch_obj_sum = 0.0
        epoch_steps = 0
        epoch_dw_sum = 0.0

        # Shuffle question indices per epoch for robust training.
        epoch_order = list(range(total_questions))
        rng.shuffle(epoch_order)

        # Exact mid-epoch resume: replay the checkpoint's saved epoch order
        # instead of re-shuffling, so we continue from the exact question the
        # interrupted run stopped at (no skipped / re-done questions).
        # `epoch_position` is the offset of this run's iteration start inside
        # the full epoch order — needed so checkpointed question positions are
        # global (survive multiple successive interruptions).
        epoch_position = 0
        if epoch == start_epoch and start_q_idx > 0:
            if saved_epoch_order and len(saved_epoch_order) == total_questions:
                epoch_order = saved_epoch_order
                q_indices = epoch_order[start_q_idx:]
                epoch_position = start_q_idx
                print(f"  Resuming epoch {epoch + 1} at position {start_q_idx} "
                      f"({len(q_indices)} questions remaining).")
            else:
                print(f"  WARNING: cannot replay saved epoch order (limit/seed "
                      f"changed?); restarting epoch {epoch + 1} from scratch.")
                q_indices = epoch_order
        else:
            q_indices = epoch_order

        pbar = tqdm(q_indices, desc=f"Epoch {epoch + 1}/{args.epochs}",
                     unit="q", dynamic_ncols=True)
        for local_idx, q_idx in enumerate(pbar):
            example = examples[q_idx]
            question = example["question"]
            golds = example_answers(example, dataset=args.dataset)
            gt = gold_titles(example, dataset=args.dataset)

            # ---- GRPO group sampling (Fix 2: all actions sampled, no greedy continuation) ----
            group_samples: list[dict] = []

            for _ in range(args.group_size):
                state = env.reset(question, golds, gt)
                first_action, old_prob = policy.sample_action(state.to_vector(), rng)
                action = first_action

                total_return = 0.0
                done = False
                while not done:
                    step_result = env.step(action)
                    total_return += step_result.reward
                    done = step_result.done
                    if not done:
                        # Sample the next action — no greedy shortcut, so the
                        # trajectory reflects the policy's true stochastic behavior
                        # and returns vary meaningfully across the group.
                        action, _ = policy.sample_action(step_result.state.to_vector(), rng)

                # Track distribution of the FIRST action (the one being updated).
                if first_action == 0:
                    action_counts["stop"] += 1
                elif first_action == 1:
                    action_counts["retrieve"] += 1
                else:
                    action_counts["reject"] += 1

                group_samples.append({
                    "state": state.to_vector(),
                    "action": first_action,
                    "old_prob": old_prob,
                    "return": total_return,
                })

            # ---- GRPO update with running baseline (Fix 1) ----
            returns = [s["return"] for s in group_samples]
            baseline.update(returns)
            advantages = baseline.advantages(returns)

            # Measure how much this step actually moved the weights. The GRPO
            # objective is ~0 by construction (advantages normalized to mean 0),
            # so max_weight_delta is the reliable "is it learning" signal.
            params_before = policy.params_snapshot()
            objective = policy.grpo_update(
                group_samples, advantages,
                lr=args.lr, clip_eps=args.clip_eps,
            )
            params_after = policy.params_snapshot()
            max_weight_delta = max(
                float(np.abs(params_after[k] - params_before[k]).max())
                for k in params_after
            )

            mean_return = float(np.mean(returns))
            ema_return = mean_return if ema_return is None else 0.9 * ema_return + 0.1 * mean_return

            epoch_return_sum += mean_return
            epoch_obj_sum += objective
            epoch_dw_sum += max_weight_delta
            epoch_steps += 1
            step_counter += 1

            # Log per-step data for visualization.
            steps_log.append({
                "step": step_counter,
                "epoch": epoch,
                "question_idx": q_idx,
                "group_return": mean_return,
                "objective": float(objective),
                "max_weight_delta": max_weight_delta,
                "action_counts": [action_counts["stop"], action_counts["retrieve"], action_counts["reject"]],
                "advantages": [float(a) for a in advantages],
                "actions": [s["action"] for s in group_samples],
            })

            pbar.set_postfix({
                "R": f"{mean_return:.3f}",
                "EMA-R": f"{ema_return:.3f}",
                "obj": f"{objective:.2e}",
                "dW": f"{max_weight_delta:.2e}",
            })

            # ---- periodic save within epoch ----
            if (local_idx + 1) % args.save_every == 0:
                action_count_history.append({
                    "question_idx": q_idx,
                    "epoch": epoch,
                    "stop": action_counts["stop"],
                    "retrieve": action_counts["retrieve"],
                    "reject": action_counts["reject"],
                })
                # Global position within the epoch order (epoch_position accounts
                # for a mid-epoch resume so repeated interruptions don't redo work).
                current_q_idx = epoch_position + local_idx + 1
                _save_checkpoint(policy, args.output, ckpt_path, epoch, current_q_idx,
                                 completed_epochs, epoch_returns, epoch_objectives,
                                 action_counts, action_count_history, epoch_order,
                                 steps_log, step_counter, time.time() - total_start_time)

                # Refresh the live training log so visualize_training.py --watch
                # can plot progress while training is still running.
                _write_train_log(args.log_out, _build_log_config(args, llm),
                                 epoch_returns, epoch_objectives, action_counts,
                                 step_counter, time.time() - total_start_time,
                                 steps_log)

                # Real-time health check (judged on weight delta + return trend,
                # NOT objective — see note above).
                _report_health(steps_log, args.save_every, action_counts, ema_return)

        # ---- end of epoch ----
        avg_return = epoch_return_sum / max(epoch_steps, 1)
        avg_obj = epoch_obj_sum / max(epoch_steps, 1)
        avg_dw = epoch_dw_sum / max(epoch_steps, 1)
        epoch_returns.append(avg_return)
        epoch_objectives.append(avg_obj)
        completed_epochs += 1

        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch + 1} done: avg_return={avg_return:.4f}, "
              f"avg_obj={avg_obj:.4f}, avg_dW={avg_dw:.2e}, time={elapsed:.1f}s")

        # Save after each epoch (resume point = next epoch, question 0).
        _save_checkpoint(policy, args.output, ckpt_path, epoch + 1, 0,
                         completed_epochs, epoch_returns, epoch_objectives,
                         action_counts, action_count_history, epoch_order,
                         steps_log, step_counter, time.time() - total_start_time)
        _write_train_log(args.log_out, _build_log_config(args, llm),
                         epoch_returns, epoch_objectives, action_counts,
                         step_counter, time.time() - total_start_time,
                         steps_log)

    # ---- training complete ----
    total_time = time.time() - total_start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Final epoch returns: {[f'{r:.4f}' for r in epoch_returns]}")
    print(f"Action distribution: {action_counts}")

    # Final training log write (same schema as the live writes above).
    _write_train_log(args.log_out, _build_log_config(args, llm), epoch_returns,
                     epoch_objectives, action_counts, step_counter,
                     total_time, steps_log)
    print(f"Training log saved to {args.log_out}")
    print(f"Policy saved to {args.output}")


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_checkpoint(
    policy: PolicyNetwork,
    policy_path: Path,
    ckpt_path: Path,
    current_epoch: int,
    current_q_idx: int,
    completed_epochs: int,
    epoch_returns: list[float],
    epoch_objectives: list[float],
    action_counts: dict[str, int],
    action_count_history: list[dict],
    epoch_order: list[int],
    steps_log: list[dict],
    step_counter: int,
    elapsed_s: float,
) -> None:
    """Atomically save a training checkpoint.

    Writes to a .tmp file first, then replaces the final path, so a crash
    during the write doesn't corrupt a valid checkpoint. Uses os.replace
    (not Path.rename) for cross-platform compatibility (Windows rename
    fails if the destination exists).

    The checkpoint carries everything needed to resume exactly:
    - policy weights (.npz),
    - progress (epoch / question position / completed epochs / elapsed),
    - the current epoch's shuffled question order (``epoch_order``) so a
      mid-epoch resume replays the same sequence,
    - the per-step log and action history, so a resumed run produces one
      continuous, uninterrupted visualization.
    """
    # Save policy weights.
    tmp_policy = policy_path.with_suffix(".tmp.npz")
    policy.save(tmp_policy)
    os.replace(tmp_policy, policy_path)

    # Save checkpoint metadata.
    ckpt = {
        "progress": {
            "current_epoch": current_epoch,
            "current_question_idx": current_q_idx,
            "completed_epochs": completed_epochs,
            "total_steps": step_counter,
            "elapsed_s": elapsed_s,
        },
        "metrics": {
            "epoch_returns": epoch_returns,
            "epoch_objectives": epoch_objectives,
            "action_counts": action_counts,
        },
        "action_count_history": action_count_history,
        "epoch_order": list(epoch_order),
        "steps_log": steps_log,
    }
    tmp_ckpt = ckpt_path.with_suffix(".tmp.ckpt.json")
    with tmp_ckpt.open("w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
    os.replace(tmp_ckpt, ckpt_path)


# ---------------------------------------------------------------------------
# Live log + health helpers
# ---------------------------------------------------------------------------

def _build_log_config(args: argparse.Namespace, llm) -> dict:
    """Config block embedded in the training log (same shape as the final log)."""
    return {
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "epochs": args.epochs,
        "group_size": args.group_size,
        "lr": args.lr,
        "clip_eps": args.clip_eps,
        "max_retrievals": args.max_retrievals,
        "initial_k": args.initial_k,
        "k_increment": args.k_increment,
        "sample_temperature": args.sample_temperature,
        "seed": args.seed,
        "model": llm.model,
    }


def _write_train_log(
    log_path: Path,
    config: dict,
    epoch_returns: list[float],
    epoch_objectives: list[float],
    action_counts: dict[str, int],
    step_counter: int,
    elapsed_s: float,
    steps_log: list[dict],
) -> None:
    """Atomically write the training log JSON (schema shared by the viz script).

    Called at every checkpoint save AND at the end of training, so the log file
    always reflects the latest progress — visualize_training.py --watch can plot
    it live while training is still running. Overwrites in place via a .tmp
    file + os.replace so a crash mid-write never corrupts the log.
    """
    log_data = {
        "config": config,
        "metrics": {
            "epoch_returns": epoch_returns,
            "epoch_objectives": epoch_objectives,
            "action_counts": action_counts,
            "total_steps": step_counter,
            "total_time_s": elapsed_s,
        },
        "steps": steps_log,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = log_path.with_name(log_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, log_path)


def _report_health(
    steps_log: list[dict],
    window: int,
    action_counts: dict[str, int],
    ema_return: float | None,
) -> None:
    """Print a compact 'is training healthy' summary from recent steps.

    The GRPO clipped objective is ~0 by construction (advantages are normalized
    to mean 0), so health is judged from two real signals:
      - max_weight_delta: how much the last updates actually moved the weights
        (zero => gradients are not flowing => nothing is being learned);
      - group-return trend over the window (EMA given as an extra anchor).
    """
    recent = steps_log[-window:]
    if not recent:
        return

    avg_dw = sum(s.get("max_weight_delta", 0.0) for s in recent) / len(recent)
    rets = [s.get("group_return", 0.0) for s in recent]
    half = len(rets) // 2
    ret_early = sum(rets[:half]) / max(half, 1)
    ret_now = sum(rets[half:]) / max(len(rets) - half, 1)

    total_ac = sum(action_counts.values()) or 1
    stop_pct = action_counts.get("stop", 0) / total_ac * 100
    retr_pct = action_counts.get("retrieve", 0) / total_ac * 100
    rej_pct = action_counts.get("reject", 0) / total_ac * 100

    flags: list[str] = []
    if avg_dw < 1e-12:
        flags.append("NOT LEARNING: weights frozen (gradient flow broken)")
    elif avg_dw < 1e-8:
        flags.append("learning slowly (tiny weight deltas)")
    if len(rets) >= 4 and ret_early > 0 and ret_now < ret_early - 0.15:
        flags.append("return trending down")
    if retr_pct < 3:
        flags.append("retrieve ratio too low (policy not exploring retrieval)")
    if stop_pct > 85:
        flags.append("stop ratio too high (try lower lr, e.g. 0.005)")

    status = "OK" if not flags else "WARNING: " + "; ".join(flags)
    print(f"\n  [Health @step {recent[-1]['step']}] "
          f"dW_avg={avg_dw:.2e}  R_early={ret_early:.3f}/R_now={ret_now:.3f}  "
          f"EMA_R={ema_return:.3f}  "
          f"Stop={stop_pct:.0f}% Retr={retr_pct:.0f}% Rej={rej_pct:.0f}%  ->  {status}")


if __name__ == "__main__":
    main()
