Measured result — LightGBM (histogram + GOSS + EFB + leaf-wise growth; 255 leaves, 500 trees, learning
rate 0.1, 16 threads, `min_data_in_leaf=0`, `min_sum_hessian_in_leaf=100`) on the Higgs and MS-LTR
benchmarks. Source: the LightGBM repository's committed `docs/Experiments.rst` (Comparison Experiment,
`LightGBM` column). Hardware: single Azure ND24s, 2×E5-2690 v4, 448GB, 16 threads.

Metrics: Higgs test **AUC** (higher is better) and **training seconds per iteration** (lower is better);
MS-LTR test **NDCG@10** (higher is better). For the speed benchmark the ranking objective is run as
`regression` for a fair cross-library comparison, per the docs.

Speed / accuracy on Higgs:

| configuration | Higgs test AUC | train s/iter |
|---|---|---|
| XGBoost histogram (step 3, `xgboost_hist`) | 0.845314 | 165.575 |
| **LightGBM** | **0.845724** | **130.094** |

Ranking quality on MS LTR:

| configuration | MS-LTR NDCG@10 |
|---|---|
| XGBoost histogram (`xgboost_hist`) | 0.496553 |
| **LightGBM** | **0.524252** |

On Higgs, GOSS + EFB drop the per-iteration training time from 165.575 s to **130.094 s** (~1.27×
faster than the histogram baseline, and ~29× faster than the exact-greedy GBM's 3794.34 s), while test
AUC edges up from 0.845314 to **0.845724** — the sampling and bundling preserve the histogram's gains.
On MS LTR (137 features), NDCG@10 rises from 0.496553 to **0.524252**, a clear ranking-quality gain.
Lower s/iter is better and higher AUC/NDCG is better; all moved the right way.

Limitation carried into the next rung: every rung so far has handled features the same way for the
*categorical* case — replacing a category with a target statistic computed from the whole training set,
which uses each example's own label when encoding its own features. The benchmarks here (Higgs, MS LTR)
are numeric, so this leakage is invisible; on a heavily categorical dataset it biases the model. That is
what the next rung attacks.
