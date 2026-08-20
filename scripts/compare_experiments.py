#!/usr/bin/env python
"""Aggregate and compare all evaluation runs across datasets x models.

Reads every artifacts/eval_results/eval_results_*.json written by run_eval.py
(the 3 datasets x 4 models matrix on the server) and produces:
  1. A markdown table of EM / F1 per (dataset, model), with retrieval recall
     and full-hit rate where recorded.
  2. Grouped bar charts (EM and F1) and a heatmap (dataset x model).
  3. Per-dataset training curves overlaid across models (from train_logs/, the
     return-EMA lines) — optional with --train.

Usage:
    python scripts/compare_experiments.py                          # eval matrix only
    python scripts/compare_experiments.py --train --window 15      # + training curves
    python scripts/compare_experiments.py --datasets triviaqa popqa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and compare all evaluation runs (dataset x model)."
    )
    parser.add_argument("--eval-dir", type=Path,
                        default=ARTIFACTS_DIR / "eval_results",
                        help="Directory of eval_results_*.json files.")
    parser.add_argument("--logs-dir", type=Path,
                        default=ARTIFACTS_DIR / "train_logs",
                        help="Directory of *_train_log.json files (for --train).")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Only include these datasets (e.g. triviaqa popqa nqswap).")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only include these model names (substring match).")
    parser.add_argument("--train", action="store_true",
                        help="Also plot per-dataset training curves across models.")
    parser.add_argument("--window", type=int, default=15,
                        help="Moving-average window for training curves.")
    parser.add_argument("--out-md", type=Path,
                        default=ARTIFACTS_DIR / "stats" / "experiment_matrix.md",
                        help="Markdown summary output path.")
    parser.add_argument("--out-bars", type=Path,
                        default=ARTIFACTS_DIR / "figures" / "compare_em_f1.png",
                        help="Grouped-bar figure output path.")
    parser.add_argument("--out-heat", type=Path,
                        default=ARTIFACTS_DIR / "figures" / "compare_heatmap.png",
                        help="Heatmap figure output path.")
    return parser.parse_args()


def _load_eval_summaries(args) -> tuple[dict, dict]:
    """Return (summaries, skipped): {(dataset, model): {metric: value}}."""
    summaries: dict[tuple[str, str], dict] = {}
    skipped: list[str] = []
    for path in sorted(args.eval_dir.glob("eval_results_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            skipped.append(f"{path.name}: unreadable ({exc})")
            continue
        if isinstance(payload, list):
            skipped.append(f"{path.name}: old list format, skip")
            continue
        config = payload.get("config", {})
        summary = payload.get("summary", {})
        dataset = config.get("dataset")
        model = config.get("model")
        if not dataset or not model:
            skipped.append(f"{path.name}: missing dataset/model in config")
            continue
        summaries[(dataset, model)] = summary
    return summaries, skipped


def _row_sort_key(ds: str) -> tuple[int, str]:
    order = ["hotpotqa", "2wikimultihopqa", "musique", "triviaqa", "popqa", "nqswap", "wikisimple"]
    return (order.index(ds) if ds in order else len(order), ds)


def _fmt_md_cell(s: dict, key: str) -> str:
    if key not in s or s[key] is None:
        return "-"
    return f"{s[key]:.3f}"


def _moving_average(values, window):
    if window <= 1 or len(values) < window:
        return list(values)
    out, acc, buf = [], 0.0, []
    for v in values:
        buf.append(v)
        acc += v
        if len(buf) > window:
            acc -= buf.pop(0)
        out.append(acc / len(buf))
    return out


def _collect_train_curves(args, datasets, models):
    """{(dataset, model): (xs, ema_returns)} from *_train_log.json files."""
    curves: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
    for path in sorted(args.logs_dir.glob("*_train_log.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        config = data.get("config", {})
        dataset, model = config.get("dataset"), config.get("model")
        if not dataset or not model:
            continue
        if dataset not in datasets or not _model_match(model, args.models):
            continue
        steps = data.get("steps", [])
        if not steps:
            continue
        returns = [s["group_return"] for s in steps]
        xs = [s["step"] for s in steps]
        curves[(dataset, model)] = (xs, _moving_average(returns, args.window))
    return curves


def _model_match(model: str, models: list[str] | None) -> bool:
    return not models or any(m in model for m in models)


def main() -> None:
    args = parse_args()
    summaries, skipped = _load_eval_summaries(args)

    # Filter by CLI.
    summaries = {
        (ds, m): s for (ds, m), s in summaries.items()
        if (not args.datasets or ds in args.datasets)
        and _model_match(m, args.models)
    }

    if not summaries:
        raise SystemExit(
            f"No usable eval_results_*.json found in {args.eval_dir}.\n"
            + ("Skipped:\n  " + "\n  ".join(skipped) if skipped else "")
        )

    datasets = sorted({ds for ds, _ in summaries}, key=_row_sort_key)
    models = sorted({m for _, m in summaries})

    # ---- markdown table (EM / F1 / recall / full-hit) ----
    lines = ["# Experiment Matrix (dataset x model)\n"]
    for metric, unit in (("avg_em", "EM"), ("avg_f1", "F1"),
                         ("avg_recall", "Recall@k"), ("full_hit_rate", "FullHit")):
        lines.append(f"## {unit}\n")
        lines.append("| Dataset | " + " | ".join(f"`{m}`" for m in models) + " |")
        lines.append("|---|" + "---|" * len(models))
        for ds in datasets:
            cells = []
            for m in models:
                s = summaries.get((ds, m), {})
                cells.append(_fmt_md_cell(s, metric))
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        lines.append("")

    # Per-run sample counts (show when runs differ).
    counts = {(ds, m): s.get("num_examples", "?") for (ds, m), s in summaries.items()}
    n_vals = set(counts.values())
    if len(n_vals) > 1:
        lines.append("## Example counts\n")
        lines.append("| Dataset | " + " | ".join(f"`{m}`" for m in models) + " |")
        lines.append("|---|" + "---|" * len(models))
        for ds in datasets:
            lines.append(f"| {ds} | " +
                         " | ".join(str(counts[(ds, m)]) for m in models) + " |")
        lines.append("")

    if skipped:
        lines.append("## Skipped files\n")
        lines.extend(f"- {s}" for s in skipped)
        lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    # Terminal print of the matrix.
    print("=" * 78)
    print(f"Experiment matrix ({len(summaries)} runs) -> {args.out_md}")
    for metric, unit in (("avg_em", "EM"), ("avg_f1", "F1")):
        print(f"\n{unit}  (dataset x model):")
        header = "  " + "".join(f"{m[:22]:>24}" for m in models)
        print(header)
        for ds in datasets:
            row = f"{ds:<14}"
            for m in models:
                s = summaries.get((ds, m), {})
                row += f"{_fmt_md_cell(s, metric):>24}"
            print(row)
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for s in skipped:
            print(f"  - {s}")
    print("=" * 78)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Grouped bars: EM and F1 side by side.
    x = range(len(datasets))
    width = 0.8 / max(len(models), 1)
    fig, axs = plt.subplots(1, 2, figsize=(max(10, 3.2 * len(datasets)), 5))
    for i, metric in enumerate(("avg_em", "avg_f1")):
        ax = axs[i]
        for j, m in enumerate(models):
            vals = [summaries.get((ds, m), {}).get(metric, 0.0) for ds in datasets]
            offset = (j - (len(models) - 1) / 2) * width
            ax.bar([v + offset for v in x], vals, width, label=m)
        ax.set_xticks(list(x))
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title("EM" if metric == "avg_em" else "F1")
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle(f"Evaluation across datasets x models ({len(models)} models)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    args.out_bars.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_bars, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grouped bars to {args.out_bars}")

    # Heatmaps: dataset x model for EM and F1.
    fig, axs = plt.subplots(1, 2, figsize=(max(10, 3.0 * len(datasets)), max(4, 0.8 * len(models) + 2)))
    for i, metric in enumerate(("avg_em", "avg_f1")):
        ax = axs[i]
        mat = [[summaries.get((ds, m), {}).get(metric, float("nan")) for m in models]
               for ds in datasets]
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=8)
        for r in range(len(datasets)):
            for c in range(len(models)):
                v = mat[r][c]
                ax.text(c, r, f"{v:.2f}" if v == v else "-",
                        ha="center", va="center", fontsize=8,
                        color="black" if v > 0.5 else "white")
        ax.set_title("EM" if metric == "avg_em" else "F1")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Heatmap: dataset x model", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(args.out_heat, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {args.out_heat}")

    # ---- training curves per dataset (across models) ----
    if args.train:
        curves = _collect_train_curves(args, datasets, models)
        if not curves:
            print("--train set but no matching *_train_log.json files found; "
                  "skipping training-curve plots.")
        for ds in datasets:
            ds_curves = {m: curves[(ds, m)] for m in models if (ds, m) in curves}
            if not ds_curves:
                continue
            plt.figure(figsize=(9, 5))
            for m, (xs, ema) in ds_curves.items():
                plt.plot(xs, ema, linewidth=2, label=m)
            plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            plt.title(f"{ds}: group-return moving average by model")
            plt.xlabel("Update step")
            plt.ylabel(f"Return (MA w={args.window})")
            plt.legend()
            plt.grid(alpha=0.3)
            out = args.out_md.parent.parent / "figures" / f"train_compare_{ds}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved training curves to {out}")

    print("Done.")


if __name__ == "__main__":
    main()
