"""
OptuML Benchmark Plotter

Reads benchmark_results.csv and generates grouped bar charts per dataset.

Usage:
    python benchmark/benchmark-plot.py
    python benchmark/benchmark-plot.py path/to/benchmark_results.csv
"""

import csv
import os
import sys
import numpy as np

QUICK_TRIALS = 20
FULL_TRIALS = 50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_records(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            def f(v):
                return float(v) if v else float("nan")
            records.append({
                "dataset": row["dataset"],
                "task": row["task"],
                "algorithm": row["algorithm"],
                "def_cv": f(row["def_cv"]),
                "def_test": f(row["def_test"]),
                "quick_cv": f(row["quick_cv"]),
                "quick_test": f(row["quick_test"]),
                "full_cv": f(row["full_cv"]),
                "full_test": f(row["full_test"]),
                "time_s": f(row["time_s"]),
            })
    return records


def plot_benchmark(all_records, output_dir):
    """Create one grouped horizontal bar chart per dataset comparing test scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError as e:
        print(f"Required library not available — skipping plots: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    from collections import defaultdict
    groups = defaultdict(list)
    for rec in all_records:
        groups[(rec["dataset"], rec["task"])].append(rec)

    palette = {
        "Default": "#4878CF",
        f"Quick ({QUICK_TRIALS} trials)": "#71D251",
        f"Full ({FULL_TRIALS} trials)": "#16B13A",
    }

    for (ds_name, task), records in groups.items():
        rows = []
        for r in records:
            rows.append({"Algorithm": r["algorithm"], "Variant": "Default",                        "Score": r["def_test"]})
            rows.append({"Algorithm": r["algorithm"], "Variant": f"Quick ({QUICK_TRIALS} trials)", "Score": r["quick_test"]})
            rows.append({"Algorithm": r["algorithm"], "Variant": f"Full ({FULL_TRIALS} trials)",   "Score": r["full_test"]})
        df = pd.DataFrame(rows)

        n = len(records)
        fig, ax = plt.subplots(figsize=(5, 0.3 * n + 1.5))

        sns.barplot(
            data=df,
            y="Algorithm",
            x="Score",
            hue="Variant",
            palette=palette,
            orient="h",
            ax=ax,
        )

        # Annotate bars with values
        for bar in ax.patches:
            w = bar.get_width()
            if not np.isnan(w) and w != 0:
                ax.annotate(
                    f"{w:.3f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=6.5,
                )

        scoring_label = "Accuracy" if task == "classification" else "R²"
        ax.set_xlabel(f"Test {scoring_label}")
        ax.set_ylabel("")
        ax.set_title(f"{ds_name} — {task.capitalize()} Test Score Comparison")
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(title="")

        fig.tight_layout()
        slug = f"{ds_name.lower().replace(' ', '_')}_{task}"
        path = os.path.join(output_dir, f"benchmark_{slug}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to:    {path}")


def fmt(val):
    if np.isnan(val):
        return "---"
    return f"{val:+.4f}" if val < 0 else f"{val:.4f}"


def save_markdown(all_records, output_dir):
    """Write one markdown table per (dataset, task) group."""
    from collections import defaultdict
    groups = defaultdict(list)
    for rec in all_records:
        groups[(rec["dataset"], rec["task"])].append(rec)

    lines = [
        "# OptuML Benchmark Results",
        "",
        f"Comparing default scikit-learn hyperparameters against OptuML optimization "
        f"({QUICK_TRIALS} quick / {FULL_TRIALS} full trials).",
        "",
        "**Columns:** `CV` = mean cross-validation score on training set; "
        "`Test` = score on held-out test set; `Time` = total wall time (default + quick + full).",
        "",
    ]

    for (ds_name, task), records in groups.items():
        scoring_label = "Accuracy" if task == "classification" else "R²"
        lines += [
            f"## {ds_name} ({task.capitalize()}) — {scoring_label}",
            "",
            f"| Algorithm | Default CV | Default Test | Quick CV | Quick Test | Full CV | Full Test | Time (s) |",
            f"|-----------|:----------:|:------------:|:--------:|:----------:|:-------:|:---------:|:--------:|",
        ]
        for r in records:
            lines.append(
                f"| {r['algorithm']} "
                f"| {fmt(r['def_cv'])} | {fmt(r['def_test'])} "
                f"| {fmt(r['quick_cv'])} | {fmt(r['quick_test'])} "
                f"| {fmt(r['full_cv'])} | {fmt(r['full_test'])} "
                f"| {r['time_s']:.1f} |"
            )
        lines.append("")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown saved to: {path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        print("Run benchmark.py first to generate results.")
        sys.exit(1)
    records = load_records(csv_path)
    plot_benchmark(records, OUTPUT_DIR)
    save_markdown(records, OUTPUT_DIR)
