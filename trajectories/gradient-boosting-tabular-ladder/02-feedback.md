Measured result — Gradient Boosting Machine with **exact (pre-sorted) split finding**, on the Higgs
benchmark. The benchmark's exact-greedy configuration (`xgboost` with `tree_method=exact`, depth 8, 500
rounds, learning rate 0.1, 16 threads) is the realization of Friedman-style exact split finding on this
suite, and is the comparison baseline in the published numbers. Source: the LightGBM repository's
committed `docs/Experiments.rst` (Comparison Experiment; the exact column is the row labeled `xgboost`).

Metrics: Higgs test **AUC** (higher is better) and **training seconds per iteration** (lower is better).
Hardware: single Azure ND24s, 2×E5-2690 v4, 448GB, 16 threads.

| configuration | Higgs test AUC | train s/iter |
|---|---|---|
| exact-greedy GBM (`xgboost`, `tree_method=exact`) | 0.839593 | **3794.34** |

The exact, pre-sorted split scan — sort every feature, evaluate every candidate threshold at every node
— sets a strong accuracy reference (AUC 0.839593) but at a punishing cost: **3794.34 seconds per
iteration** on Higgs (10.5M rows, 28 features). That per-iteration wall-clock is the number the next
rung has to beat; the bottleneck is the exhaustive split search, not the loss or the model class.
(Provenance note: in `docs/Experiments.rst`, `xgboost_exact` is the slow exact-greedy setting kept as a
reference and not re-run on every update of the benchmark — the value above is its published figure.)
