#!/usr/bin/env python
"""Visualize GRPO training dynamics logged by scripts/train_policy.py.

Reads the training log JSON and plots:
  1. Per-step group return with a moving average (learning curve).
  2. Clipped-objective per step.
  3. First-action distribution (stop/retrieve/reject) over training.
  4. Per-step weight delta (max |dW|) — the reliable "is it learning" signal,
     since the GRPO objective is ~0 by construction.

The log file is rewritten at every checkpoint, so --watch can plot progress
live while training is still running (e.g. in a second terminal).

Usage:
    # One-shot render of a finished or in-progress run
    python scripts/visualize_training.py --dataset triviaqa --model llama3:8b

    # Live monitoring: re-render the PNG every 10 s until interrupted (Ctrl-C)
    python scripts/visualize_training.py --dataset triviaqa --model llama3:8b --watch 10
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _default_log_path(dataset: str, model: str | None) -> Path:
    """Auto-derive the log path from dataset and model, matching train_policy.py."""
    if model is None:
        from rag_learning.config import LLM_MODEL
        model = LLM_MODEL
    from rag_learning.config import model_slug
    slug = model_slug(model)
    return ARTIFACTS_DIR / "train_logs" / f"{dataset}_{slug}_train_log.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GRPO training curves.")
    parser.add_argument("--dataset", default="hotpotqa",
                        help="Dataset name (hotpotqa, 2wikimultihopqa, popqa, nqswap, triviaqa, ...).")
    parser.add_argument("--model", default=None,
                        help="LLM model name. Auto-detected from .env if not set.")
    parser.add_argument("--log", type=Path, default=None,
                        help="Training log JSON path. Auto-derived if not set.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path. Defaults to "
                             "artifacts/figures/grpo_training_<dataset>_<model>.png.")
    parser.add_argument("--window", type=int, default=15, help="Moving-average window.")
    parser.add_argument("--watch", type=int, default=None, metavar="SECONDS",
                        help="Live mode: re-render the plot every N seconds while "
                             "training runs, until interrupted (Ctrl-C).")
    return parser.parse_args()


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    acc = 0.0
    buf = deque()
    for v in values:
        buf.append(v)
        acc += v
        if len(buf) > window:
            acc -= buf.popleft()
        out.append(acc / len(buf))
    return out


def _render(args: argparse.Namespace) -> bool:
    """Plot one frame from the current log. Returns True if data was rendered."""
    if not args.log.exists():
        return False

    with args.log.open("r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not steps:
        return False

    config = data.get("config", {})
    xs = [s["step"] for s in steps]
    returns = [s["group_return"] for s in steps]
    objectives = [s["objective"] for s in steps]
    dw = [s.get("max_weight_delta", 0.0) for s in steps]
    counts = [s.get("action_counts", [0, 0, 0]) for s in steps]

    ma = moving_average(returns, args.window)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))

    # --- 1. return curve ---
    axs[0][0].plot(xs, returns, color="#b0c4de", linewidth=1, label="group return")
    axs[0][0].plot(xs, ma, color="#1f4e79", linewidth=2, label=f"moving avg (w={args.window})")
    axs[0][0].set_title("Mean Group Return per Update")
    axs[0][0].set_xlabel("Update step (one per training question)")
    axs[0][0].set_ylabel("Return")
    axs[0][0].legend()

    # --- 2. objective ---
    axs[0][1].plot(xs, objectives, color="#c0504d", linewidth=1)
    axs[0][1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axs[0][1].set_title("GRPO Clipped Objective per Update (~0 by construction)")
    axs[0][1].set_xlabel("Update step")
    axs[0][1].set_ylabel("Objective")

    # --- 3. weight delta (the real learning signal) ---
    axs[1][0].plot(xs, dw, color="#7f7f7f", linewidth=1, label="max |dW| per step")
    axs[1][0].plot(xs, moving_average(dw, args.window), color="#c00000", linewidth=2,
                   label=f"moving avg (w={args.window})")
    axs[1][0].set_yscale("symlog", linthresh=1e-10)
    axs[1][0].set_title("Weight Delta per Update (frozen at 0 => no learning)")
    axs[1][0].set_xlabel("Update step")
    axs[1][0].set_ylabel("max |ΔW|")
    axs[1][0].legend()

    # --- 4. action distribution (stacked moving avg) ---
    stop = [c[0] for c in counts]
    retr = [c[1] for c in counts]
    rej = [c[2] for c in counts]
    axs[1][1].plot(xs, moving_average(stop, args.window), label="stop", color="#4472c4", linewidth=2)
    axs[1][1].plot(xs, moving_average(retr, args.window), label="retrieve", color="#70ad47", linewidth=2)
    axs[1][1].plot(xs, moving_average(rej, args.window), label="reject", color="#ed7d31", linewidth=2)
    axs[1][1].set_title(f"Sampled First-Action Mix (moving avg, w={args.window})")
    axs[1][1].set_xlabel("Update step")
    axs[1][1].set_ylabel("Avg count per group")
    axs[1][1].legend()

    # Mark epoch boundaries precisely from the per-step "epoch" field
    # (robust to resume — steps continue numbering across interruptions).
    boundaries = [i for i in range(1, len(steps)) if steps[i]["epoch"] != steps[i - 1]["epoch"]]
    for b in boundaries:
        for row in axs:
            for ax in row:
                ax.axvline(xs[b], color="gray", linewidth=0.7, linestyle=":")

    subtitle = (
        f"GRPO training  |  {config.get('dataset', args.dataset)} / {config.get('model', '?')}  |  "
        f"{config.get('limit')} q x {config.get('epochs')} epochs, "
        f"G={config.get('group_size')}, lr={config.get('lr')}  |  "
        f"step {xs[-1]}/{config.get('total_steps', xs[-1])}  |  "
        f"last MA return={ma[-1]:.3f}"
    )
    plt.suptitle(subtitle, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    if args.log is None:
        args.log = _default_log_path(args.dataset, args.model)
    if args.out is None:
        from rag_learning.config import model_slug
        slug = model_slug(args.model)
        args.out = ARTIFACTS_DIR / "figures" / f"grpo_training_{args.dataset}_{slug}.png"

    if not args.log.exists():
        raise FileNotFoundError(
            f"Training log not found: {args.log}. Run train_policy.py first."
        )

    if args.watch:
        print(f"Live monitoring {args.log} every {args.watch}s — Ctrl-C to stop.")
        print(f"Plotting to {args.out}")
        while True:
            try:
                rendered = _render(args)
                if rendered:
                    print(f"[{time.strftime('%H:%M:%S')}] updated {args.out.name} "
                          f"({args.log.stat().st_mtime:.0f} mtime)")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] waiting for first logged step...")
            except Exception as exc:  # keep monitoring alive across transient issues
                print(f"[{time.strftime('%H:%M:%S')}] render error: {exc}")
            time.sleep(args.watch)
    else:
        if not _render(args):
            raise ValueError(f"No per-step data in {args.log}. Re-run train_policy.py.")
        print(f"Saved training curve to {args.out}")


if __name__ == "__main__":
    main()
