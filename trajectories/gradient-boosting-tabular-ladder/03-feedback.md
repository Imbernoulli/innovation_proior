Measured result — XGBoost with **histogram (weighted-quantile) split finding** on the Higgs benchmark
(`xgboost_hist`: `tree_method=hist`, `grow_policy=lossguide`, 255 leaves, 500 rounds, learning rate 0.1,
16 threads). Source: the LightGBM repository's committed `docs/Experiments.rst` (Comparison Experiment,
`xgboost_hist` column). Hardware: single Azure ND24s, 2×E5-2690 v4, 448GB, 16 threads.

Metrics: Higgs test **AUC** (higher is better) and **training seconds per iteration** (lower is better).

| configuration | Higgs test AUC | train s/iter |
|---|---|---|
| exact-greedy GBM (step 2, `xgboost` exact) | 0.839593 | 3794.34 |
| **XGBoost histogram** (`xgboost_hist`) | **0.845314** | **165.575** |

Replacing the exact, pre-sorted scan with the Hessian-weighted quantile histogram cuts the
per-iteration training time from 3794.34 s to **165.575 s** — a ~22.9× speedup — while the test AUC
*rises* from 0.839593 to **0.845314**, the second-order regularized gain criterion buying accuracy at the
same time it buys speed. Lower s/iter and higher AUC are both better; both improved.

Limitation carried into the next rung: even with the histogram, each iteration still accumulates
gradient/Hessian sums over **all** the data and **all** the features — the per-iteration cost remains
proportional to (#data) × (#features), which is what the next rung attacks.
