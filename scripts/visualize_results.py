#!/usr/bin/env python
"""Visualize the results produced by scripts/run_eval.py.

This script does NOT run the model itself. It reads the JSON file written by
run_eval.py (default: artifacts/eval_results.json) and produces summary tables
and a distribution histogram, so the visualization always reflects the exact
run_eval.py run.

Usage:
    python scripts/run_eval.py --split validation --limit 50 --top-k 5
    python scripts/visualize_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the evaluation results produced by run_eval.py."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to the JSON results written by run_eval.py (--results-out). "
        "Defaults to artifacts/eval_results_<current-model>.json.",
    )
    parser.add_argument(
        "--hist-out",
        type=Path,
        default=None,
        help="Histogram output path. Defaults to artifacts/results_hist_<model>.png.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="Stats markdown output. Defaults to artifacts/result_stats_<model>.md.",
    )
    return parser.parse_args()


def main() -> None:
    from rag_learning.config import model_slug

    args = parse_args()

    # Default input: the results file for the model currently set in .env.
    if args.results is None:
        args.results = ARTIFACTS_DIR / f"eval_results_{model_slug()}.json"

    if not args.results.exists():
        raise FileNotFoundError(
            f"Results file not found: {args.results}. "
            f"Run `python scripts/run_eval.py --split validation --limit 50 --top-k 5` first."
        )

    with args.results.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    # Backward compatibility: older runs saved a bare list of per-example dicts,
    # while run_eval.py now saves {"config", "summary", "results"}.
    if isinstance(payload, list):
        results = payload
        config = {}
        summary = {}
        print(
            "WARNING: eval_results.json is in the OLD list format (not produced by the "
            "current run_eval.py). Model/config info is unavailable. "
            "Re-run `python scripts/run_eval.py` to regenerate it."
        )
    else:
        config = payload.get("config", {})
        summary = payload.get("summary", {})
        results = payload.get("results", [])

    if not results:
        raise ValueError(f"No per-example results found in {args.results}.")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]

    model = config.get("model", "unknown")

    # Name outputs after the model recorded in the results file, so different
    # models produce distinct artifacts instead of overwriting each other.
    slug = model_slug(model)
    if args.hist_out is None:
        args.hist_out = ARTIFACTS_DIR / f"results_hist_{slug}.png"
    if args.stats_out is None:
        args.stats_out = ARTIFACTS_DIR / f"result_stats_{slug}.md"

    avg_em = summary.get("avg_em", sum(em_scores) / len(em_scores))
    avg_f1 = summary.get("avg_f1", sum(f1_scores) / len(f1_scores))

    counter = Counter(r.get("category", "incorrect_answer") for r in results)
    total = len(results)
    correct = counter.get("correct", 0)
    rejected = counter.get("rejected_unknown", 0)
    incorrect = counter.get("incorrect_answer", 0)

    print("=" * 80)
    print("Evaluation Summary (from run_eval.py results):")
    print(f"  Results file: {args.results}")
    print(f"  Model:        {model}")
    print(f"  Split/limit/top_k: {config.get('split')}/{config.get('limit')}/{config.get('top_k')}")
    print(f"  Total examples: {total}")
    print(f"  Average EM: {avg_em:.3f} ({avg_em * 100:.1f}%)")
    print(f"  Average F1: {avg_f1:.3f} ({avg_f1 * 100:.1f}%)")
    print()
    print("Category breakdown:")
    print(f"  Correct (EM > 0):      {correct:>3} / {total} ({correct / total * 100:.1f}%)")
    print(f"  Rejected (unknown):    {rejected:>3} / {total} ({rejected / total * 100:.1f}%)")
    print(f"  Incorrect answer:      {incorrect:>3} / {total} ({incorrect / total * 100:.1f}%)")
    print("=" * 80)

    # save stats markdown for report
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_out.open("w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        f.write(f"**测试模型 (LLM): `{model}`**\n\n")
        f.write(f"- Model: `{model}`\n")
        f.write(f"- Dataset split: `{config.get('split')}`\n")
        f.write(f"- Number of examples: {config.get('limit', total)}\n")
        f.write(f"- Top-k retrieval: {config.get('top_k')}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Average EM | **{avg_em:.3f}** ({avg_em * 100:.1f}%) |\n")
        f.write(f"| Average F1 | **{avg_f1:.3f}** ({avg_f1 * 100:.1f}%) |\n\n")
        f.write("## Result Category\n\n")
        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        f.write(f"| Correct (EM > 0) | {correct} | {correct / total * 100:.1f}% |\n")
        f.write(f"| Rejected (unknown) | {rejected} | {rejected / total * 100:.1f}% |\n")
        f.write(f"| Incorrect answer | {incorrect} | {incorrect / total * 100:.1f}% |\n")
    print(f"Saved stats to {args.stats_out}")

    # plot histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    axs[0].hist(em_scores, bins=bins, edgecolor="black")
    axs[0].set_title("Exact Match (EM) Distribution")
    axs[0].set_xlabel("EM Score")
    axs[0].set_ylabel("Count")
    axs[0].set_xticks(bins)
    axs[1].hist(f1_scores, bins=bins, edgecolor="black")
    axs[1].set_title("F1 Score Distribution")
    axs[1].set_xlabel("F1 Score")
    axs[1].set_ylabel("Count")
    axs[1].set_xticks(bins)
    plt.suptitle(
        f"LLM: {model}  |  split={config.get('split')}, "
        f"n={total}, top_k={config.get('top_k')}  |  "
        f"EM={avg_em * 100:.1f}%, F1={avg_f1 * 100:.1f}%",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(args.hist_out, dpi=150, bbox_inches="tight")
    print(f"Saved histogram to {args.hist_out}")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
