"""
Benchmark all supported algorithms on classification (Iris) and regression
(Diabetes) datasets using AlgorithmBenchmark, then plot ranking charts and
save results to CSV files.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

from optuml import AlgorithmBenchmark

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRIALS = 30
RANDOM_STATE = 42
OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

X_clf, y_clf = load_iris(return_X_y=True)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE
)

X_reg, y_reg = load_diabetes(return_X_y=True)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_ranking(summary_df, title, metric_label, output_path, color):
    """Horizontal bar chart of algorithm scores, best at the top."""
    df = summary_df.dropna(subset=["best_score"]).sort_values("best_score")

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)))
    bars = ax.barh(df["algorithm"], df["best_score"], color=color, edgecolor="white")

    # Annotate each bar with the score value
    for bar, score in zip(bars, df["best_score"]):
        ax.text(
            bar.get_width() + (df["best_score"].max() * 0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.margins(x=0.12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


def run_benchmark(task, X_train, y_train, label, metric_label, color):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    bench = AlgorithmBenchmark(
        task=task,
        n_trials=N_TRIALS,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    bench.fit(X_train, y_train)

    summary = bench.summary()
    print(summary.to_string(index=False))

    csv_path = OUTPUT_DIR / f"benchmark_{task}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    plot_path = OUTPUT_DIR / f"benchmark_{task}.png"
    plot_ranking(summary, label, metric_label, plot_path, color)

    print(f"\n  Best algorithm : {bench.best_algorithm_}")
    print(f"  Best CV score  : {bench.best_score_:.4f}")

    return bench, summary


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

clf_bench, clf_summary = run_benchmark(
    task="classification",
    X_train=X_clf_train,
    y_train=y_clf_train,
    label="Classification benchmark — Iris dataset (CV accuracy)",
    metric_label="CV Accuracy",
    color="steelblue",
)

reg_bench, reg_summary = run_benchmark(
    task="regression",
    X_train=X_reg_train,
    y_train=y_reg_train,
    label="Regression benchmark — Diabetes dataset (CV R²)",
    metric_label="CV R²",
    color="darkorange",
)

# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  Summary")
print(f"{'='*60}")
print(f"Classification winner : {clf_bench.best_algorithm_}  "
      f"(accuracy={clf_bench.best_score_:.4f})")
print(f"Regression winner     : {reg_bench.best_algorithm_}  "
      f"(R²={reg_bench.best_score_:.4f})")
print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
