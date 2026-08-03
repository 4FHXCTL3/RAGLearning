#!/usr/bin/env python
"""GRPO policy training for adaptive RAG (CLARITY-RAG Framework Module 2).

Trains a PolicyNetwork on a multi-hop QA dataset using GRPO (Group Relative
Policy Optimization). Supports checkpoint save/resume so training can survive
interruptions (power loss, crash, etc.) and epochs can be extended without
retraining from scratch.

Usage:
    # Train from scratch (3 epochs, 100 examples, 2WikiMultihopQA)
    python scripts/train_policy.py --dataset 2wikimultihopqa --limit 100 --epochs 3

    # Resume interrupted training
    python scripts/train_policy.py --dataset 2wikimultihopqa --limit 100 --epochs 3 --resume

    # Extend training to more epochs (e.g. 5 instead of 3)
    python scripts/train_policy.py --dataset 2wikimultihopqa --limit 100 --epochs 5 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Enable offline mode for HuggingFace datasets (use local cache only).
# This avoids network timeouts in restricted environments.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import DEFAULT_INDEX_PATH, model_slug
from rag_learning.data import gold_titles
from rag_learning.policy import PolicyNetwork, group_advantages

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
                        choices=["hotpotqa", "2wikimultihopqa", "musique"],
                        help="Dataset to train on.")
    parser.add_argument("--split", default="train",
                        help="Dataset split (train / validation / test).")
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
                        help="Resume from existing checkpoint if available.")
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
    rng = np.random.default_rng(args.seed)

    # --- resolve paths ---
    llm = ChatLLM()
    slug = model_slug(llm.model)

    if args.output is None:
        args.output = ARTIFACTS_DIR / "policies" / f"{args.dataset}_{slug}_policy.npz"
    if args.log_out is None:
        args.log_out = ARTIFACTS_DIR / "train_logs" / f"{args.dataset}_{slug}_train_log.json"

    ckpt_path = checkpoint_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_out.parent.mkdir(parents=True, exist_ok=True)

    # --- checkpoint state (used for resume) ---
    start_epoch = 0
    start_q_idx = 0
    epoch_returns: list[float] = []
    epoch_objectives: list[float] = []
    action_counts: dict[str, int] = {"stop": 0, "retrieve": 0, "reject": 0}
    completed_epochs = 0

    # --- load or init policy ---
    policy: PolicyNetwork
    if args.resume and args.output.exists():
        policy = PolicyNetwork.load(args.output)
        if ckpt_path.exists():
            with ckpt_path.open("r", encoding="utf-8") as f:
                ckpt = json.load(f)
            progress = ckpt.get("progress", {})
            start_epoch = progress.get("current_epoch", 0)
            start_q_idx = progress.get("current_question_idx", 0)
            completed_epochs = progress.get("completed_epochs", 0)
            epoch_returns = ckpt.get("metrics", {}).get("epoch_returns", [])
            epoch_objectives = ckpt.get("metrics", {}).get("epoch_objectives", [])
            action_counts = ckpt.get("metrics", {}).get("action_counts",
                                                         {"stop": 0, "retrieve": 0, "reject": 0})
            print(f"Resumed from checkpoint: epoch={start_epoch}/{args.epochs}, "
                  f"question_idx={start_q_idx}, completed_epochs={completed_epochs}")
        else:
            print(f"Warning: --resume set but no checkpoint found at {ckpt_path}. "
                  f"Starting from scratch.")
    else:
        policy = PolicyNetwork(seed=args.seed)
        print("Initialized new policy network.")

    # Check if already fully trained.
    if completed_epochs >= args.epochs:
        print(f"Training already completed: {completed_epochs} >= {args.epochs} epochs. "
              f"Use --epochs {completed_epochs + 1} (or higher) to continue training.")
        return

    # --- build / load index ---
    if args.index is not None and args.index.exists():
        index = BM25Index.load(args.index)
    else:
        idx_path = _build_index(args.dataset, args.split, args.limit)
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

    total_start_time = time.time()
    step_counter = 0  # total GRPO steps across all epochs
    steps_log: list[dict] = []  # per-step data for visualization

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_return_sum = 0.0
        epoch_obj_sum = 0.0
        epoch_steps = 0

        # Shuffle question indices per epoch for robust training.
        q_indices = list(range(total_questions))
        rng.shuffle(q_indices)

        # If resuming mid-epoch, skip already-completed questions.
        if epoch == start_epoch and start_q_idx > 0:
            q_indices = [i for i in q_indices if i >= start_q_idx]

        pbar = tqdm(q_indices, desc=f"Epoch {epoch + 1}/{args.epochs}",
                     unit="q", dynamic_ncols=True)
        for local_idx, q_idx in enumerate(pbar):
            example = examples[q_idx]
            question = example["question"]
            gold_answer = example["answer"]
            gt = gold_titles(example, dataset=args.dataset)

            # ---- GRPO group sampling ----
            group_samples: list[dict] = []

            for _ in range(args.group_size):
                state = env.reset(question, gold_answer, gt)
                action, old_prob = policy.sample_action(state.to_vector(), rng)

                step_result = env.step(action)
                total_return = step_result.reward

                # Greedy continuation until terminal.
                while not step_result.done:
                    greedy_action = int(np.argmax(policy.action_probs(step_result.state.to_vector())))
                    step_result = env.step(greedy_action)
                    total_return += step_result.reward

                # Track action distribution.
                if action == 0:
                    action_counts["stop"] += 1
                elif action == 1:
                    action_counts["retrieve"] += 1
                else:
                    action_counts["reject"] += 1

                group_samples.append({
                    "state": state.to_vector(),
                    "action": action,
                    "old_prob": old_prob,
                    "return": total_return,
                })

            # ---- GRPO update ----
            returns = [s["return"] for s in group_samples]
            advantages = group_advantages(returns)
            objective = policy.grpo_update(
                group_samples, advantages,
                lr=args.lr, clip_eps=args.clip_eps,
            )

            epoch_return_sum += np.mean(returns)
            epoch_obj_sum += objective
            epoch_steps += 1
            step_counter += 1

            # Log per-step data for visualization.
            steps_log.append({
                "step": step_counter,
                "epoch": epoch,
                "question_idx": q_idx,
                "group_return": float(np.mean(returns)),
                "objective": float(objective),
                "action_counts": [action_counts["stop"], action_counts["retrieve"], action_counts["reject"]],
                "advantages": [float(a) for a in advantages],
                "actions": [s["action"] for s in group_samples],
            })

            pbar.set_postfix({
                "R": f"{np.mean(returns):.3f}",
                "obj": f"{objective:.4f}",
            })

            # ---- periodic save within epoch ----
            if (local_idx + 1) % args.save_every == 0:
                _save_checkpoint(policy, args.output, ckpt_path, epoch, local_idx + 1,
                                 completed_epochs, epoch_returns, epoch_objectives,
                                 action_counts)

        # ---- end of epoch ----
        avg_return = epoch_return_sum / max(epoch_steps, 1)
        avg_obj = epoch_obj_sum / max(epoch_steps, 1)
        epoch_returns.append(avg_return)
        epoch_objectives.append(avg_obj)
        completed_epochs += 1

        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch + 1} done: avg_return={avg_return:.4f}, "
              f"avg_obj={avg_obj:.4f}, time={elapsed:.1f}s")

        # Save after each epoch.
        _save_checkpoint(policy, args.output, ckpt_path, epoch + 1, 0,
                         completed_epochs, epoch_returns, epoch_objectives,
                         action_counts)

    # ---- training complete ----
    total_time = time.time() - total_start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Final epoch returns: {[f'{r:.4f}' for r in epoch_returns]}")
    print(f"Action distribution: {action_counts}")

    # Save training log.
    log_data = {
        "config": {
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
        },
        "metrics": {
            "epoch_returns": epoch_returns,
            "epoch_objectives": epoch_objectives,
            "action_counts": action_counts,
            "total_steps": step_counter,
            "total_time_s": total_time,
        },
        "steps": steps_log,
    }
    with args.log_out.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
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
) -> None:
    """Atomically save a training checkpoint.

    Writes to a .tmp file first, then replaces the final path, so a crash
    during the write doesn't corrupt a valid checkpoint. Uses os.replace
    (not Path.rename) for cross-platform compatibility (Windows rename
    fails if the destination exists).
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
        },
        "metrics": {
            "epoch_returns": epoch_returns,
            "epoch_objectives": epoch_objectives,
            "action_counts": action_counts,
        },
    }
    tmp_ckpt = ckpt_path.with_suffix(".tmp.ckpt.json")
    with tmp_ckpt.open("w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
    os.replace(tmp_ckpt, ckpt_path)


if __name__ == "__main__":
    main()
