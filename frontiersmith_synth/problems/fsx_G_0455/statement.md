# Fraud Feature Sieve: Selecting Signals for a Frozen Scorer

## Story

A payments company screens card transactions for fraud. For each transaction its risk
team logs a **wide** table of engineered features — velocity counters, device
fingerprints, geo-mismatch flags, merchant-category indicators, and many more. Most of
these columns are **noise**: only a handful genuinely separate fraud from legitimate
activity, and because the training log is finite, plenty of pure-noise columns show a
**deceptive in-sample correlation** with the fraud label.

Downstream, a **frozen, fixed scoring model** is retrained every night on *whatever
feature subset the risk team hands it*. The frozen model is a **standardized
nearest-centroid classifier**: it standardizes each chosen column on the training log,
computes a "fraud" centroid and a "legitimate" centroid, and labels a new transaction
by whichever centroid is nearer in the selected subspace. Every extra irrelevant column
injects distance noise, so handing the model **all** columns is badly hurt by the curse
of dimensionality — yet dropping a genuinely informative column loses signal.

Your job is to write the **feature-selection heuristic**: given only the training log,
choose the subset of columns the frozen model should be retrained on. You are scored on
the model's accuracy on **held-out transactions you never see**, geometric-mean-averaged
over a bank of fraud datasets.

## The frozen model (fixed; identical for every subset `S`)

Fit on the training rows restricted to the chosen columns `S`:

- For each column `j` in `S`: `mu_j`, `sd_j` = mean and standard deviation over the
  **training** rows (if `sd_j < 1e-9`, use `sd_j = 1`).
- For each class `c` in `{0, 1}`: `centroid_c[j]` = mean over the training rows of class
  `c` of the standardized value `(x_j - mu_j) / sd_j`.

Predict a transaction `x` by choosing the class `c` that minimizes
`sum_{j in S} ( (x_j - mu_j)/sd_j - centroid_c[j] )^2` (ties go to the majority training
class). Accuracy is the fraction of held-out transactions predicted correctly.

You do **not** run this model — the evaluator does. Your program only outputs the column
subset.

## Input (stdin): ONE JSON object — the PUBLIC training log

```json
{
  "name": "fraud102",
  "n_features": 150,
  "n_train": 350,
  "X_train": [[f0, f1, ..., f149], ...],   // n_train rows of floats
  "y_train": [0, 1, 0, ...]                // n_train fraud labels (1 = fraud)
}
```

The held-out transactions, the identities of the truly-informative columns, and all
scoring anchors are hidden from your program.

## Output (stdout): ONE JSON object — the chosen columns

```json
{"features": [3, 17, 42, ...]}
```

`features` must be a **non-empty list of distinct integers**, each in `[0, n_features)`,
of length at most `n_features`. Order does not matter. Any violation — a crash, a
timeout, non-JSON, `null`, a wrong type, an out-of-range or duplicated index, or an empty
list — scores **0.0** on that dataset.

## Objective — MAXIMIZE

For each dataset the evaluator computes, on the same held-out set:

- `a_all`    — frozen-model accuracy using **all** columns (the weak baseline),
- `a_oracle` — frozen-model accuracy using the true **planted informative** columns,
- `a_cand`   — frozen-model accuracy using **your** subset,

and normalizes

```
r = clamp( 0.1 + 0.9 * (a_cand - a_all) / max(a_oracle - a_all, 0.02),  0, 1 )
```

(a valid selection is floored at `0.02`, so a legal-but-poor subset stays distinct from
an invalid one, which scores exactly `0.0`).

- Keeping every column reproduces `a_all` → `r ≈ 0.1`.
- Matching the planted informative columns approaches `a_oracle` → `r ≈ 1.0`.
- The finite log's spurious correlations keep even good filters strictly below `1.0` on
  most datasets, so there is genuine headroom.

The reported **Ratio** is the **geometric mean** of `r` across the whole dataset bank — a
heuristic must select well *everywhere*, not just on the easy datasets. There is no single
dominant strategy: univariate filtering, cross-validated budget tuning, redundancy/stability
analysis, and forward search all trade off signal breadth against added distance noise.

## Determinism & isolation

Scoring is fully deterministic (all datasets are seeded; the model has no randomness or
wall-time dependence). Your program is run **isolated** in a fresh sandboxed subprocess
and only ever sees the public training log.
