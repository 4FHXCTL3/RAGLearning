#!/usr/bin/env python
"""Quick smoke-test for the GRPO training loop.

Runs 1 epoch on a small number of examples and reports whether the policy
network weights actually changed — the #1 symptom of the previous broken
training was that `mean_objective` hovered at ~1e-16 and weights stayed
frozen.

Usage:
    python scripts/check_training.py --dataset 2wikimultihopqa --group-size 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.data import example_answers, gold_titles, load_examples
from rag_learning.embed import Embedder
from rag_learning.environment import RAGEnv, RewardWeights
from rag_learning.index import BM25Index
from rag_learning.llm import ChatLLM
from rag_learning.policy import PolicyNetwork, RunningBaseline
from rag_learning.state import StateEncoder

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def build_index(dataset: str, split: str, limit: int | None) -> Path:
    from rag_learning.data import iter_context_paragraphs
    from rag_learning.index import BM25Index

    idx_path = ARTIFACTS_DIR / "indexes" / f"bm25_index_{dataset}_{split}.pkl"
    if idx_path.exists():
        print(f"  Index exists at {idx_path}, loading...")
        return idx_path

    print(f"  Building BM25 index for {dataset} ({split}, limit={limit})...")
    examples = load_examples(split=split, limit=limit, dataset=dataset)
    documents = list(iter_context_paragraphs(examples, dataset=dataset))
    index = BM25Index(documents)
    index.save(idx_path)
    print(f"  Indexed {len(documents)} paragraphs from {len(examples)} examples.")
    return idx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick training smoke-test")
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--model", default=None,
                        help="Ollama model name. Defaults to LLM_MODEL from .env.")
    parser.add_argument("--offline", action="store_true",
                        help="Force HuggingFace datasets offline mode (local cache only).")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--limit", type=int, default=3,
                        help="Number of questions to test on")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=Path, default=None,
                        help="Path to BM25 index (uses existing file, avoids download)")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-retrievals", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    print("=" * 60)
    print("  GRPO Training Smoke Test")
    print("=" * 60)
    print(f"  dataset={args.dataset}, group_size={args.group_size}, limit={args.limit}")
    print(f"  lr={args.lr}, max_retrievals={args.max_retrievals}")
    print()

    rng = np.random.default_rng(args.seed)

    # --- load model + index ---
    llm = ChatLLM(model=args.model)
    embedder = Embedder()
    encoder = StateEncoder(embedder=embedder)

    if args.index and args.index.exists():
        idx_path = args.index
        print(f"  Loading index from {idx_path}...")
    else:
        idx_path = build_index(args.dataset, args.split, args.limit)
    index = BM25Index.load(idx_path)
    print(f"  Index loaded: {len(index.documents)} documents")
    print()

    # --- load examples ---
    # Use the index's original dataset to load matching examples.
    examples = load_examples(split=args.split, limit=args.limit,
                             dataset=args.dataset)
    print(f"  Loaded {len(examples)} examples")
    print()

    # --- init env + policy ---
    weights = RewardWeights()
    env = RAGEnv(
        index=index,
        llm=llm,
        encoder=encoder,
        weights=weights,
        max_retrievals=args.max_retrievals,
        group_size=args.group_size,
    )
    policy = PolicyNetwork(seed=args.seed)

    # --- snapshot initial weights ---
    w1_before = policy.w1.copy()
    w2_before = policy.w2.copy()

    # --- run 1 epoch ---
    baseline = RunningBaseline(alpha=0.9)
    total_obj = 0.0
    total_return = 0.0
    action_dist = {"stop": 0, "retrieve": 0, "reject": 0}
    gradients_nonzero = 0
    n_steps = 0

    print(f"{'Step':>5} {'Q_idx':>6} {'R_mean':>8} {'Obj':>10} {'Actions':>18} "
          f"{'Adv':>20}")
    print("-" * 70)

    for q_idx in range(len(examples)):
        example = examples[q_idx]
        question = example["question"]
        golds = example_answers(example, dataset=args.dataset)
        gt = gold_titles(example, dataset=args.dataset)

        # --- sample group (full sampling, no greedy) ---
        group_samples: list[dict] = []
        for _ in range(args.group_size):
            state = env.reset(question, golds, gt)
            first_action, old_prob, first_cache = policy.sample_action(state.to_vector(), rng)
            action = first_action

            total_return_traj = 0.0
            done = False
            while not done:
                step_result = env.step(action)
                total_return_traj += step_result.reward
                done = step_result.done
                if not done:
                    action, _prob, _cache = policy.sample_action(
                        step_result.state.to_vector(), rng
                    )

            if first_action == 0:
                action_dist["stop"] += 1
            elif first_action == 1:
                action_dist["retrieve"] += 1
            else:
                action_dist["reject"] += 1

            group_samples.append({
                "state": state.to_vector(),
                "action": first_action,
                "old_prob": old_prob,
                "return": total_return_traj,
                "cache": first_cache,
            })

        # --- GRPO update ---
        returns = [s["return"] for s in group_samples]
        baseline.update(returns)
        advantages = baseline.advantages(returns)
        objective, _max_dw = policy.grpo_update(
            group_samples, advantages,
            lr=args.lr, clip_eps=0.2,
        )

        total_obj += objective
        total_return += float(np.mean(returns))
        n_steps += 1

        # Check if weights actually changed.
        delta_w1 = np.abs(policy.w1 - w1_before).max()
        delta_w2 = np.abs(policy.w2 - w2_before).max()
        if delta_w1 > 1e-12 or delta_w2 > 1e-12:
            gradients_nonzero += 1

        actions_str = ",".join(str(s["action"]) for s in group_samples)
        print(f"{n_steps:>5} {q_idx:>6} {np.mean(returns):>8.4f} "
              f"{objective:>10.6f} {actions_str:>18} "
              f"{str([f'{a:.2f}' for a in advantages]):>20}")

    # --- results ---
    print("=" * 70)
    print()
    print("  RESULTS")
    print("  -------")
    avg_obj = total_obj / max(n_steps, 1)
    avg_ret = total_return / max(n_steps, 1)
    print(f"  Mean objective:        {avg_obj:.8f}")
    print(f"  Mean return:           {avg_ret:.4f}")
    print(f"  Steps with gradient:   {gradients_nonzero} / {n_steps}")
    print(f"  Action distribution:   {action_dist}")
    print()

    # --- verdict ---
    print("  VERDICT")
    print("  -------")

    if gradients_nonzero == 0:
        print("  FAIL: Policy weights never changed (gradients are zero).")
        print("     The RunningBaseline or grpo_update still isn't working.")
    elif avg_obj < 1e-6 and avg_obj > -1e-6:
        print("  WEAK: Weights changed but mean_objective is near zero.")
        print(f"     avg_obj = {avg_obj:.2e}.  Update magnitude suspiciously small.")
    else:
        print(f"  PASS: mean_objective = {avg_obj:.6f}, weights changed.")
        print("     The training loop is producing meaningful gradient updates.")

    all_actions = [s["action"] for s in group_samples]
    if len(set(all_actions)) == 1 and action_dist.get("reject", 0) > 0:
        print("  WARNING: All actions are REJECT. Reward may still favor rejection.")
    elif len(set(all_actions)) == 1:
        print("  WARNING: All actions identical -- limited exploration.")
    else:
        print("  Actions are diverse -- the policy is exploring.")

    print()

    # Save a brief log for reference.
    log_out = ARTIFACTS_DIR / "train_logs" / "smoke_test_result.json"
    log_data = {
        "config": {k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()},
        "results": {
            "mean_objective": avg_obj,
            "mean_return": avg_ret,
            "gradients_nonzero": gradients_nonzero,
            "action_distribution": action_dist,
        },
    }
    log_out.parent.mkdir(parents=True, exist_ok=True)
    with log_out.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"  Log saved to {log_out}")


if __name__ == "__main__":
    main()
