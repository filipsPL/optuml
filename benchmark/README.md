# Benchmark results

- [tables](results/benchmark_results.md)
- [csv source](results/benchmark_results.csv) / [per-trial trajectories](results/benchmark_trials.csv)
- [plots](results/)

## Running

```bash
python benchmark/benchmark.py                 # full run: all datasets x all supported algorithms
python benchmark/benchmark-plot.py            # plots + markdown from results/*.csv

# quick smoke run on a subset
python benchmark/benchmark.py --datasets Iris Friedman1 --algorithms SVC SVR \
    --quick-trials 5 --full-trials 10 --output-dir /tmp/bench
python benchmark/benchmark-plot.py /tmp/bench/benchmark_results.csv
```

## Plots

- `benchmark_<dataset>.png`: absolute held-out score (default → quick → full) next to the
  gain over default (CV gain bars, test gain diamonds).
- `convergence_<dataset>.png`: per algorithm, best-so-far CV score minus default CV score vs trial number.
- `convergence_summary.png`: median/IQR of that gain across algorithms, per dataset, i.e. how many
  trials are needed to collect most of the gain.
- `gain_heatmap.png`: held-out test gain, algorithm × dataset.

## Notes

- The benchmarked algorithms are taken from `ClassifierOptimizer` / `RegressorOptimizer` `_get_supported_algorithms()`, so optional packages that are installed are included automatically.
- Datasets: Iris, Wine, Breast Cancer, a synthetic 3-class problem (`make_classification` with label noise; the real ones are near-saturated, which hides tuning gains), Diabetes and Friedman #1.
- The default-hyperparameter baseline is scaled exactly like OptuML (StandardScaler pipeline for scale-sensitive algorithms, target standardization for some regressors) and uses the same shuffled, seeded CV splitter, so the comparison measures hyperparameter tuning only.
- CV gain is optimistically biased (it is the best of many trials on the same folds); the held-out test gain is the honest number.
- The results in `benchmark/results/` predate these changes (unscaled baselines, hand-written 16-algorithm list, no trial trajectories) until the benchmark is re-run.
