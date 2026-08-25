from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_learning.config import DEFAULT_INDEX_PATH
from rag_learning.data import gold_titles

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GRPO policy (framework module 2).")
    parser.add_argument("--dataset", default="hotpotqa", help="Dataset: hotpotqa or 2wikimultihopqa.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=120, help="Number of training questions.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=3, help="GRPO group size G.")
    parser.add_argument("--max-retrievals", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "policies" / "policy.npz")
    parser.add_argument("--log-out", type=Path, default=ARTIFACTS_DIR / "train_logs" / "grpo_train_log.json")
    return parser.parse_args()


def rollout(env, policy, state_vec, rng):
    """Sample one action from the policy at the initial state, execute it to a
    terminal step, and return (action, old_prob, episode_return, cache).

    We optimize the first decision (stop/retrieve/reject). If the action is
    RETRIEVE, the episode continues greedily (argmax) to termination so the
    group return reflects the full consequence of that first choice.

    The returned ``cache`` is the forward-cache from the initial state; the
    GRPO update reuses it to skip a redundant MLP forward.
    """
    from rag_learning.environment import ACTION_RETRIEVE

    action, old_prob, cache = policy.sample_action(state_vec, rng)
    step = env.step(action)
    total = step.reward
    # Continue non-terminal episodes greedily until done (bounded by budget).
    guard = 0
    while not step.done and guard < 5:
        nxt_probs = policy.action_probs(step.state.to_vector())
        # Greedy but avoid infinite retrieve loops; budget forces stop anyway.
        nxt_action = int(np.argmax(nxt_probs))
        step = env.step(nxt_action)
        total += step.reward
        guard += 1
    return action, old_prob, total, cache


def main() -> None:
    from rag_learning.data import load_examples
    from rag_learning.environment import RAGEnv
    from rag_learning.index import BM25Index
    from rag_learning.llm import ChatLLM
    from rag_learning.policy import PolicyNetwork, group_advantages
    from rag_learning.state import StateEncoder

    args = parse_args()
    if not args.index.exists():
        raise FileNotFoundError(f"Index not found: {args.index}. Run `python scripts/build_index.py --dataset {args.dataset} --split {args.split} --limit {args.limit}` first.")

    examples = load_examples(split=args.split, limit=args.limit, dataset=args.dataset)
    index = BM25Index.load(args.index)
    encoder = StateEncoder()
    llm = ChatLLM()
    policy = PolicyNetwork(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    env = RAGEnv(
        index=index,
        llm=llm,
        encoder=encoder,
        group_size=args.group_size,
        max_retrievals=args.max_retrievals,
    )

    history: list[dict] = []
    steps: list[dict] = []  # per-question update records for a smooth curve
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        ep_returns: list[float] = []
        ep_obj: list[float] = []
        for example in tqdm(examples, desc=f"epoch {epoch}"):
            question = example["question"]
            gold = example["answer"]
            golds = gold_titles(example, dataset=args.dataset)

            # Observe the shared initial state once (expensive: LLM + embed).
            init_state = env.reset(question, gold, golds)
            state_vec = init_state.to_vector()

            samples: list[dict] = []
            rewards: list[float] = []
            actions: list[int] = []
            for _ in range(args.group_size):
                # Re-reset per group member so each rollout starts identically.
                env.reset(question, gold, golds)
                action, old_prob, ret, cache = rollout(env, policy, state_vec, rng)
                samples.append({"state": state_vec, "action": action,
                                "old_prob": old_prob, "cache": cache})
                rewards.append(ret)
                actions.append(action)

            adv = group_advantages(rewards)
            obj, _dW = policy.grpo_update(samples, adv, lr=args.lr)
            group_ret = float(np.mean(rewards))
            ep_returns.append(group_ret)
            ep_obj.append(obj)

            global_step += 1
            steps.append({
                "step": global_step,
                "epoch": epoch,
                "group_return": group_ret,
                "objective": obj,
                # first-action distribution this group (stop/retrieve/reject counts)
                "action_counts": [actions.count(0), actions.count(1), actions.count(2)],
            })

        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_obj = float(np.mean(ep_obj)) if ep_obj else 0.0
        print(f"[epoch {epoch}] mean_return={mean_ret:.4f}  mean_objective={mean_obj:.4f}")
        history.append({"epoch": epoch, "mean_return": mean_ret, "mean_objective": mean_obj})

    policy.save(args.out)
    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    with args.log_out.open("w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args) | {"index": str(args.index), "out": str(args.out),
                                    "log_out": str(args.log_out)},
            "history": history,
            "steps": steps,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved policy to {args.out}")
    print(f"Saved training log to {args.log_out}")


if __name__ == "__main__":
    main()
