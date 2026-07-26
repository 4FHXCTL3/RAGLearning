#!/usr/bin/env python
"""Visualize GRPO training dynamics logged by scripts/train_grpo.py.

Reads artifacts/grpo_train_log.json (the "steps" array) and plots:
  1. Per-step group return with a moving average (learning curve).
  2. Clipped-objective per step.
  3. First-action distribution (stop/retrieve/reject) over training.

Usage:
    python scripts/train_grpo.py ...
    python scripts/visualize_training.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GRPO training curves.")
    parser.add_argument("--log", type=Path, default=ARTIFACTS_DIR / "train_logs" / "grpo_train_log.json")
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "figures" / "grpo_training_curve.png")
    parser.add_argument("--window", type=int, default=15, help="Moving-average window.")
    return parser.parse_args()


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    acc = 0.0
    from collections import deque
    buf = deque()
    for v in values:
        buf.append(v)
        acc += v
        if len(buf) > window:
            acc -= buf.popleft()
        out.append(acc / len(buf))
    return out


def main() -> None:
    args = parse_args()
    if not args.log.exists():
        raise FileNotFoundError(f"Training log not found: {args.log}. Run train_grpo.py first.")

    with args.log.open("r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not steps:
        raise ValueError(
            f"No per-step data in {args.log}. Re-run train_grpo.py (it now logs 'steps')."
        )

    config = data.get("config", {})
    model = "GRPO policy"
    xs = [s["step"] for s in steps]
    returns = [s["group_return"] for s in steps]
    objectives = [s["objective"] for s in steps]
    counts = [s.get("action_counts", [0, 0, 0]) for s in steps]
    n_epochs = int(config.get("epochs", 0) or (steps[-1]["epoch"] if steps else 0))

    ma = moving_average(returns, args.window)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))

    # --- 1. return curve ---
    axs[0].plot(xs, returns, color="#b0c4de", linewidth=1, label="group return")
    axs[0].plot(xs, ma, color="#1f4e79", linewidth=2, label=f"moving avg (w={args.window})")
    axs[0].set_title("Mean Group Return per Update")
    axs[0].set_xlabel("Update step (one per training question)")
    axs[0].set_ylabel("Return")
    axs[0].legend()

    # --- 2. objective ---
    axs[1].plot(xs, objectives, color="#c0504d", linewidth=1)
    axs[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axs[1].set_title("GRPO Clipped Objective per Update")
    axs[1].set_xlabel("Update step")
    axs[1].set_ylabel("Objective")

    # --- 3. action distribution (stacked) over training ---
    stop = [c[0] for c in counts]
    retr = [c[1] for c in counts]
    rej = [c[2] for c in counts]
    stop_ma = moving_average(stop, args.window)
    retr_ma = moving_average(retr, args.window)
    rej_ma = moving_average(rej, args.window)
    axs[2].plot(xs, stop_ma, label="stop", color="#4472c4", linewidth=2)
    axs[2].plot(xs, retr_ma, label="retrieve", color="#70ad47", linewidth=2)
    axs[2].plot(xs, rej_ma, label="reject", color="#ed7d31", linewidth=2)
    axs[2].set_title(f"Sampled First-Action Mix (moving avg, w={args.window})")
    axs[2].set_xlabel("Update step")
    axs[2].set_ylabel("Avg count per group")
    axs[2].legend()

    # Mark epoch boundaries on all panels.
    if n_epochs > 1:
        per_epoch = len(steps) / n_epochs
        for e in range(1, n_epochs):
            for ax in axs:
                ax.axvline(e * per_epoch, color="gray", linewidth=0.7, linestyle=":")

    subtitle = (
        f"GRPO training  |  {config.get('limit')} q x {config.get('epochs')} epochs, "
        f"G={config.get('group_size')}, lr={config.get('lr')}  |  "
        f"final MA return={ma[-1]:.3f}"
    )
    plt.suptitle(subtitle, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curve to {args.out}")


if __name__ == "__main__":
    main()
