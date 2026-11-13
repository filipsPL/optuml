"""
OptuML Benchmark Plotter

Reads the CSVs written by ``benchmark.py`` and generates plots and a markdown
report that show both the absolute score of each algorithm and the *gain* that
hyperparameter optimization brings over the default hyperparameters.

Outputs (in the directory of the results CSV):

- ``benchmark_<dataset>.png``   -- per dataset, two panels sharing the algorithm axis:
  absolute test score (default -> quick -> full) and gain over default
  (CV gain as bars, held-out test gain as diamonds).
- ``convergence_<dataset>.png`` -- per dataset, small multiples: best-so-far CV score
  minus the default CV score, as a function of the trial number (needs
  ``benchmark_trials.csv``).
- ``convergence_summary.png``   -- median (and IQR) of that gain across algorithms,
  one panel per dataset: how many trials are needed to collect most of the gain.
- ``gain_heatmap.png``          -- test-score gain, algorithm x dataset.
- ``benchmark_results.md``      -- tables with absolute scores and gains.

Usage:
    python benchmark/benchmark-plot.py
    python benchmark/benchmark-plot.py path/to/benchmark_results.csv [path/to/benchmark_trials.csv]
"""

import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

DEFAULT_QUICK_TRIALS = 20  # used only for CSVs written before the budget was recorded
DEFAULT_FULL_TRIALS = 50

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Colors (validated reference palette; light surface). Quick/full are two steps of
# one hue because they are ordered levels of the same thing (more trials).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
C_DEFAULT = "#8a8984"
C_QUICK = "#86b6ef"
C_FULL = "#2a78d6"
C_GAIN = "#2a78d6"
C_LOSS = "#e34948"
C_NEUTRAL = "#f0efec"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _f(v):
    try:
        return float(v) if v not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def load_records(csv_path):
    records = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rec = {"dataset": row["dataset"], "task": row["task"], "algorithm": row["algorithm"]}
            for k in ("def_cv", "def_test", "quick_cv", "quick_test", "full_cv", "full_test", "time_s", "full_n_trials"):
                rec[k] = _f(row.get(k))
            q, fl = _f(row.get("quick_trials")), _f(row.get("full_trials"))
            rec["quick_trials"] = int(q) if not math.isnan(q) else DEFAULT_QUICK_TRIALS
            rec["full_trials"] = int(fl) if not math.isnan(fl) else DEFAULT_FULL_TRIALS
            rec["gain_cv"] = rec["full_cv"] - rec["def_cv"]
            rec["gain_test"] = rec["full_test"] - rec["def_test"]
            rec["gain_test_quick"] = rec["quick_test"] - rec["def_test"]
            records.append(rec)
    return records


def load_trials(csv_path):
    """{(dataset, task, algorithm): (trials ndarray, value ndarray, best_so_far ndarray)}"""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    raw = defaultdict(list)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            raw[(row["dataset"], row["task"], row["algorithm"])].append(
                (int(row["trial"]), _f(row["value"]), _f(row["best_so_far"]))
            )
    out = {}
    for key, rows in raw.items():
        rows.sort()
        arr = np.array(rows, dtype=float)
        out[key] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def group_by_dataset(records):
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["dataset"], rec["task"])].append(rec)
    return groups


def metric_label(task):
    return "accuracy" if task == "classification" else "R²"


def slug(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def budgets(records):
    return records[0]["quick_trials"], records[0]["full_trials"]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "text.color": INK,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _grid(ax, axis="x"):
    getattr(ax, f"{axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _robust_floor(values, task, pad=0.03):
    """Lower x-limit that does not let one catastrophic default (e.g. R² = -40)
    squash everyone else. Points below it are drawn at the floor with a marker."""
    v = np.asarray([x for x in values if not math.isnan(x)])
    if v.size == 0:
        return 0.0
    q1, q3 = np.percentile(v, [25, 75])
    floor = q1 - 3 * max(q3 - q1, 0.02)
    if task == "regression":
        floor = max(floor, -1.0)
    lo = max(v.min(), floor)
    return lo - pad * max(v.max() - lo, 0.05)


def _sym_limit(values, cap):
    """Symmetric axis limit around 0: fits the data, but a single extreme gain
    (default that failed badly) is clipped and labelled instead of flattening the rest."""
    v = np.abs([x for x in values if not math.isnan(x)])
    if v.size == 0:
        return 0.05
    robust = 3 * np.percentile(v, 90) if v.size >= 4 else v.max()
    return float(min(max(min(v.max(), robust) * 1.15, 0.01), cap))


# ---------------------------------------------------------------------------
# Figure 1: absolute score + gain, per dataset
# ---------------------------------------------------------------------------


def plot_dataset(ds_name, task, records, out_dir, plt):
    quick_n, full_n = budgets(records)
    recs = sorted(records, key=lambda r: (-np.nan_to_num(r["full_test"], nan=-np.inf), r["algorithm"]))
    n = len(recs)
    y = np.arange(n)[::-1]
    names = [r["algorithm"] for r in recs]

    fig, (ax_abs, ax_gain) = plt.subplots(
        1, 2, figsize=(10, 0.27 * n + 1.9), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # --- left: absolute held-out test score -------------------------------------
    d = np.array([r["def_test"] for r in recs])
    q = np.array([r["quick_test"] for r in recs])
    f = np.array([r["full_test"] for r in recs])
    lo = _robust_floor(np.concatenate([d, q, f]), task)
    hi_vals = np.concatenate([d, q, f])
    hi = np.nanmax(hi_vals) if np.isfinite(hi_vals).any() else 1.0
    hi = hi + 0.03 * max(hi - lo, 0.05)

    def clip(a):
        return np.where(np.isnan(a), np.nan, np.maximum(a, lo))

    dc, qc, fc = clip(d), clip(q), clip(f)
    for yi, a, b in zip(y, dc, fc):
        if not (math.isnan(a) or math.isnan(b)):
            ax_abs.plot([a, b], [yi, yi], color=GRID, linewidth=2.5, solid_capstyle="round", zorder=1)
    ax_abs.scatter(dc, y, s=34, facecolors=SURFACE, edgecolors=C_DEFAULT, linewidths=1.6, zorder=3, label="Default")
    ax_abs.scatter(qc, y, s=30, color=C_QUICK, edgecolors=SURFACE, linewidths=1.0, zorder=4,
                   label=f"OptuML, {quick_n} trials")
    ax_abs.scatter(fc, y, s=42, color=C_FULL, edgecolors=SURFACE, linewidths=1.0, zorder=5,
                   label=f"OptuML, {full_n} trials")
    for arr in (d, q, f):  # off-scale markers
        for yi, v in zip(y, arr):
            if not math.isnan(v) and v < lo:
                ax_abs.annotate(f"{v:.2f}", (lo, yi), xytext=(3, 4), textcoords="offset points",
                                fontsize=6.5, color=INK_2)
    ax_abs.set_xlim(lo, hi)
    ax_abs.set_yticks(y, names)
    ax_abs.set_ylim(-0.7, n - 0.3)
    ax_abs.set_xlabel(f"Held-out test {metric_label(task)}  (higher is better)")
    ax_abs.set_title("Absolute score", loc="left")
    _grid(ax_abs)

    # --- right: gain over the default ---------------------------------------------
    gcv = np.array([r["gain_cv"] for r in recs])
    gte = np.array([r["gain_test"] for r in recs])
    lim = _sym_limit(np.concatenate([gcv, gte]), cap=0.3 if task == "classification" else 0.5)
    gcv_c, gte_c = np.clip(gcv, -lim, lim), np.clip(gte, -lim, lim)
    colors = [C_GAIN if (not math.isnan(g) and g >= 0) else C_LOSS for g in gcv]
    ax_gain.barh(y, np.nan_to_num(gcv_c), height=0.62, color=colors, alpha=0.85, edgecolor=SURFACE, linewidth=1,
                 label=f"CV gain ({full_n} trials, optimistic)")
    ax_gain.scatter(gte_c, y, marker="D", s=26, color=INK, edgecolors=SURFACE, linewidths=0.8, zorder=4,
                    label="Held-out test gain")
    # off-scale values: CV gain labelled above the row, test gain below it
    for yi, rcv, rte in zip(y, gcv, gte):
        for rv, dy, tag in ((rcv, 4, "CV"), (rte, -11, "test")):
            if not math.isnan(rv) and abs(rv) > lim:
                ax_gain.annotate(f"{tag} {rv:+.2f}", (np.sign(rv) * lim, yi), xytext=(-3 if rv > 0 else 3, dy),
                                 textcoords="offset points", ha="right" if rv > 0 else "left",
                                 fontsize=6.5, color=INK_2)
    ax_gain.axvline(0, color=INK_2, linewidth=0.9)
    ax_gain.set_xlim(-lim, lim)
    ax_gain.set_xlabel(f"Optimized − default ({metric_label(task)})   ← worse | better →")
    ax_gain.set_title("Gain from optimization", loc="left")
    ax_gain.tick_params(axis="y", length=0)
    _grid(ax_gain)

    improved = int(np.nansum(gte > 1e-9))
    worse = int(np.nansum(gte < -1e-9))
    fig.suptitle(
        f"{ds_name} ({task}) — test gain: median {np.nanmedian(gte):+.3f}, "
        f"{improved} improved / {worse} worse / {n - improved - worse} unchanged",
        x=0.01, ha="left", fontsize=10.5, color=INK,
    )
    handles = ax_abs.get_legend_handles_labels()[0] + ax_gain.get_legend_handles_labels()[0]
    labels = ax_abs.get_legend_handles_labels()[1] + ax_gain.get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7.5, handletextpad=0.3,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.4 / fig.get_figheight(), 1, 1))
    path = os.path.join(out_dir, f"benchmark_{slug(ds_name)}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved:     {path}")


# ---------------------------------------------------------------------------
# Figure 2: convergence small multiples, per dataset
# ---------------------------------------------------------------------------


def _gain_curve(rec, trials, full_n):
    """Best-so-far CV gain over default, forward-filled to the full budget."""
    t, v, best = trials
    x = np.arange(1, full_n + 1)
    curve = np.full(full_n, np.nan)
    for ti, b in zip(t.astype(int), best):
        if 1 <= ti <= full_n:
            curve[ti - 1] = b
    # forward-fill (trials missing because of timeout / parameter-free cap keep the last best)
    last = np.nan
    for i in range(full_n):
        if math.isnan(curve[i]):
            curve[i] = last
        else:
            last = curve[i]
    return x, curve - rec["def_cv"], v - rec["def_cv"], t


def plot_convergence(ds_name, task, records, trials, out_dir, plt):
    from matplotlib.ticker import MaxNLocator

    quick_n, full_n = budgets(records)
    items = []
    for r in records:
        tr = trials.get((ds_name, task, r["algorithm"]))
        if tr is None or math.isnan(r["def_cv"]):
            continue
        x, curve, pts, t = _gain_curve(r, tr, full_n)
        items.append((r["algorithm"], x, curve, pts, t))
    if not items:
        return
    items.sort(key=lambda it: -np.nan_to_num(it[2][-1], nan=-np.inf))

    finals = np.array([it[2][-1] for it in items])
    firsts = np.array([np.nanmin(it[2]) if np.isfinite(it[2]).any() else np.nan for it in items])
    hi = np.nanpercentile(finals, 95) if np.isfinite(finals).any() else 0.05
    lo = np.nanpercentile(firsts, 5) if np.isfinite(firsts).any() else -0.05
    hi = max(hi, 0.01) * 1.25
    lo = min(lo, -0.01)
    lo = max(lo, -2 * hi)  # a terrible first trial should not flatten every panel

    ncols = 6
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 1.45 * nrows + 0.9), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(items):]:
        ax.set_visible(False)

    for ax, (name, x, curve, pts, t) in zip(axes.flat, items):
        ax.axhline(0, color=C_DEFAULT, linewidth=1, linestyle=(0, (3, 2)))
        ax.axvline(quick_n, color=GRID, linewidth=1)
        ax.scatter(t, np.clip(pts, lo, hi), s=5, color=C_DEFAULT, alpha=0.45, linewidths=0, zorder=2)
        pos = np.where(curve >= 0, curve, 0)
        ax.fill_between(x, 0, pos, step="post", color=C_GAIN, alpha=0.15, linewidth=0)
        ax.step(x, np.clip(curve, lo, hi), where="post", color=C_FULL, linewidth=1.8, zorder=3)
        final = curve[-1]
        ax.set_title(name, fontsize=7.5, loc="left", pad=2)
        if not math.isnan(final):
            ax.text(0.97, 0.06, f"{final:+.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color=INK if final >= 0 else C_LOSS, fontweight="bold")
        ax.tick_params(labelsize=6.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        _grid(ax, "y")

    axes.flat[0].set_ylim(lo, hi)
    axes.flat[0].set_xlim(1, full_n)
    for ax in axes[-1]:
        ax.set_xlabel("trial", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Δ CV {metric_label(task)}", fontsize=7)
    fig.suptitle(
        f"{ds_name} — best-so-far CV score minus default CV score  "
        f"(dashed = default, grey dots = individual trials, vertical line = {quick_n} trials)",
        x=0.01, ha="left", fontsize=9.5, color=INK,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f"convergence_{slug(ds_name)}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved:     {path}")


def plot_convergence_summary(groups, trials, out_dir, plt):
    from matplotlib.ticker import MaxNLocator

    panels = []
    for (ds_name, task), records in groups.items():
        quick_n, full_n = budgets(records)
        curves = []
        for r in records:
            tr = trials.get((ds_name, task, r["algorithm"]))
            if tr is None or math.isnan(r["def_cv"]):
                continue
            curves.append(_gain_curve(r, tr, full_n)[1])
        if curves:
            panels.append((ds_name, task, quick_n, full_n, np.vstack(curves)))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 2.8), squeeze=False)
    for ax, (ds_name, task, quick_n, full_n, C) in zip(axes[0], panels):
        x = np.arange(1, full_n + 1)
        with np.errstate(all="ignore"), _quiet_nan():
            med = np.nanmedian(C, axis=0)
            q1, q3 = np.nanpercentile(C, 25, axis=0), np.nanpercentile(C, 75, axis=0)
        ax.axhline(0, color=C_DEFAULT, linewidth=1, linestyle=(0, (3, 2)))
        ax.axvline(quick_n, color=GRID, linewidth=1)
        ax.fill_between(x, q1, q3, step="post", color=C_GAIN, alpha=0.18, linewidth=0, label="IQR across algorithms")
        ax.step(x, med, where="post", color=C_FULL, linewidth=2, label="median")
        # trials needed to reach 90% of the final median gain
        final = med[-1]
        if np.isfinite(final) and final > 0:
            reach = int(np.argmax(med >= 0.9 * final)) + 1
            ax.annotate(f"90% of gain\nby trial {reach}", (reach, med[reach - 1]), xytext=(8, -22),
                        textcoords="offset points", fontsize=7, color=INK_2,
                        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.7))
        ax.set_title(f"{ds_name}", loc="left")
        ax.set_xlabel("trial")
        ax.set_ylabel(f"Δ CV {metric_label(task)} vs default")
        ax.set_xlim(1, full_n)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _grid(ax, "y")
    axes[0][0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Gain over default hyperparameters as trials accumulate (best-so-far inner-CV score)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved:     {path}")


class _quiet_nan:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


# ---------------------------------------------------------------------------
# Figure 3: test-gain heatmap, algorithm x dataset
# ---------------------------------------------------------------------------


def plot_gain_heatmap(groups, out_dir, plt):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    by_task = defaultdict(dict)
    for (ds_name, task), records in groups.items():
        by_task[task][ds_name] = {r["algorithm"]: r["gain_test"] for r in records}
    tasks = [t for t in ("classification", "regression") if t in by_task]
    if not tasks:
        return

    cmap = LinearSegmentedColormap.from_list("gain", [C_LOSS, C_NEUTRAL, C_GAIN])
    cmap.set_bad(SURFACE)
    heights = []
    tables = []
    for task in tasks:
        datasets = list(by_task[task])
        algos = sorted({a for d in by_task[task].values() for a in d})
        M = np.array([[by_task[task][d].get(a, np.nan) for d in datasets] for a in algos])
        order = np.argsort(-np.nan_to_num(np.nanmean(np.where(np.isnan(M), np.nan, M), axis=1), nan=-np.inf)) \
            if M.size else []
        M = M[order]
        algos = [algos[i] for i in order]
        tables.append((task, datasets, algos, M))
        heights.append(len(algos))

    width = 2.2 + 1.15 * max(len(t[1]) for t in tables)
    fig, axes = plt.subplots(1, len(tables), figsize=(width * len(tables), 0.24 * max(heights) + 1.6),
                             squeeze=False)
    for ax, (task, datasets, algos, M) in zip(axes[0], tables):
        lim = _sym_limit(M.ravel(), cap=0.15 if task == "classification" else 0.3)
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:+.3f}" if abs(v) >= 5e-4 else "0", ha="center", va="center", fontsize=6.5,
                            color=SURFACE if abs(v) > 0.6 * lim else INK)
        ax.set_xticks(range(len(datasets)), datasets, rotation=20, ha="right")
        ax.set_yticks(range(len(algos)), algos, fontsize=7.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(datasets)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(algos)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"{task.capitalize()} — Δ test {metric_label(task)}", loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, extend="both")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Held-out test gain of OptuML (full budget) over default hyperparameters",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    path = os.path.join(out_dir, "gain_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved:     {path}")


def make_plots(records, trials, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available — skipping plots: {e}")
        return
    _style(plt)
    os.makedirs(out_dir, exist_ok=True)
    groups = group_by_dataset(records)
    for (ds_name, task), recs in groups.items():
        plot_dataset(ds_name, task, recs, out_dir, plt)
        if trials:
            plot_convergence(ds_name, task, recs, trials, out_dir, plt)
    if trials:
        plot_convergence_summary(groups, trials, out_dir, plt)
    else:
        print("No benchmark_trials.csv — skipping convergence plots (re-run benchmark.py to produce it).")
    plot_gain_heatmap(groups, out_dir, plt)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def fmt(val, signed=False):
    if math.isnan(val):
        return "---"
    return f"{val:+.4f}" if signed or val < 0 else f"{val:.4f}"


def _trials_to_fraction(rec, trials, frac=0.9):
    tr = trials.get((rec["dataset"], rec["task"], rec["algorithm"]))
    if tr is None or math.isnan(rec["def_cv"]):
        return None
    _, curve, _, _ = _gain_curve(rec, tr, rec["full_trials"])
    final = curve[-1]
    if not np.isfinite(final) or final <= 0:
        return None
    return int(np.argmax(curve >= frac * final)) + 1


def save_markdown(records, trials, out_dir):
    groups = group_by_dataset(records)
    quick_n, full_n = budgets(records)
    lines = [
        "# OptuML Benchmark Results",
        "",
        f"Default scikit-learn hyperparameters vs. OptuML optimization ({quick_n} quick / {full_n} full trials).",
        "",
        "- `CV` = mean inner cross-validation score on the training set (for OptuML this is the best trial, "
        "so it is optimistically biased); `Test` = score on the held-out test set.",
        "- `ΔCV` / `ΔTest` = full-budget OptuML minus default (positive = optimization helped).",
        "- `90% @` = trial at which the best-so-far CV score first reached 90% of its final gain.",
        "",
        "## Summary",
        "",
        "| Dataset | Task | Algorithms | Improved (test) | Worse (test) | Median ΔTest | Best default (test) | Best OptuML (test) |",
        "|---------|------|:----------:|:---------------:|:------------:|:------------:|:-------------------:|:------------------:|",
    ]
    for (ds_name, task), recs in groups.items():
        g = np.array([r["gain_test"] for r in recs])
        bd = max(recs, key=lambda r: np.nan_to_num(r["def_test"], nan=-np.inf))
        bf = max(recs, key=lambda r: np.nan_to_num(r["full_test"], nan=-np.inf))
        lines.append(
            f"| {ds_name} | {task} | {len(recs)} | {int(np.nansum(g > 1e-9))} | {int(np.nansum(g < -1e-9))} "
            f"| {fmt(np.nanmedian(g), True)} | {fmt(bd['def_test'])} ({bd['algorithm']}) "
            f"| {fmt(bf['full_test'])} ({bf['algorithm']}) |"
        )
    lines.append("")

    for (ds_name, task), recs in groups.items():
        recs = sorted(recs, key=lambda r: -np.nan_to_num(r["full_test"], nan=-np.inf))
        lines += [
            f"## {ds_name} ({task.capitalize()}) — {metric_label(task)}",
            "",
            f"![{ds_name}](benchmark_{slug(ds_name)}.png)",
            "",
            "| Algorithm | Default CV | Default Test | Quick Test | Full CV | Full Test | ΔCV | ΔTest | 90% @ | Time (s) |",
            "|-----------|:----------:|:------------:|:----------:|:-------:|:---------:|:---:|:-----:|:-----:|:--------:|",
        ]
        for r in recs:
            reach = _trials_to_fraction(r, trials) if trials else None
            lines.append(
                f"| {r['algorithm']} | {fmt(r['def_cv'])} | {fmt(r['def_test'])} | {fmt(r['quick_test'])} "
                f"| {fmt(r['full_cv'])} | {fmt(r['full_test'])} | {fmt(r['gain_cv'], True)} "
                f"| {fmt(r['gain_test'], True)} | {reach if reach else '---'} | {r['time_s']:.1f} |"
            )
        lines.append("")
        if trials:
            lines += [f"![{ds_name} convergence](convergence_{slug(ds_name)}.png)", ""]

    if trials:
        lines += ["## Convergence across algorithms", "", "![convergence summary](convergence_summary.png)", ""]
    lines += ["## Test gain heatmap", "", "![gain heatmap](gain_heatmap.png)", ""]

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "benchmark_results.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Markdown saved: {path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS_DIR, "benchmark_results.csv")
    trials_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(csv_path), "benchmark_trials.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        print("Run benchmark.py first to generate results.")
        sys.exit(1)
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    records = load_records(csv_path)
    trials = load_trials(trials_path)
    make_plots(records, trials, out_dir)
    save_markdown(records, trials, out_dir)
