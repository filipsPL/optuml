# Benchmark results

- [tables](results/benchmark_results.md)
- [csv source](results/benchmark_results.csv)
- [plots](results/)

## Notes

- The benchmarked algorithms are taken from `ClassifierOptimizer` / `RegressorOptimizer` `_get_supported_algorithms()`, so optional packages that are installed are included automatically.
- The default-hyperparameter baseline is now scaled exactly like OptuML (StandardScaler pipeline for scale-sensitive algorithms, target standardization for some regressors) and uses the same shuffled, seeded CV splitter, so the comparison measures hyperparameter tuning only.
- The results in `benchmark/results/` predate these changes (unscaled baselines, hand-written algorithm list) until the benchmark is re-run.
