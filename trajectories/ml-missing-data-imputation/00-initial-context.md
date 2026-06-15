## Research question

A tabular data matrix arrives with holes in it: entries are `NaN`, scattered across columns and rows. Almost nothing downstream tolerates a `NaN` — the gradient-boosted model that will be scored cannot put one in a split, a distance between two rows is undefined the moment one coordinate is absent. So the matrix has to be turned into a finite-valued completion of the same shape. The single thing being designed is the **imputer itself**: how feature dependencies are exploited, how the fills are iterated or refined, and how completed values are produced from a matrix that still contains `NaN`. Everything around it — the datasets, the corruption, the scaling, the downstream model — is fixed.

## Prior art before the first rung (completion lineage)

The first rung — unconditional mean imputation — is the floor a long line of completion methods reacts to. These are the ideas that precede the ladder; the fixed substrate below is the contract every one of them must fill.

- **Listwise deletion (complete-case analysis, the reflex).** Drop every row with a hole and run on what's left. Unbiased under MCAR, but with `d` columns each missing at rate `p` the fraction of complete rows is `(1 − p)^d`, which at `d = 30`, `p = 0.2` is `≈ 0.0012` — it throws away nearly the whole table — and it still cannot score a new row that has a hole. Gap: collapses with width and never solves test-time missingness.
- **Unconditional mean / median imputation (Little & Rubin).** One fixed number per column written into all its holes. Cheap, deterministic, and the column mean is the least-squares constant predictor of the column. Gap: it is *univariate* — it uses nothing about the rest of the row, so it ignores every inter-feature correlation, deflates the column's variance, and pulls correlations toward zero.
- **Hot-deck / neighbour donation.** Fill a hole by copying a value from a "similar" record. The right instinct — use the row's observed coordinates — but it needs a distance that is undefined when coordinates are missing, and a literal single donor is jumpy.
- **Regression imputation (Buck 1960).** Regress the incomplete variable on the others and predict the holes — finally exploits correlations. Gap: the predictors are themselves incomplete, and a single regression with one variable at a time is circular when missingness spans many columns.
- **EM under a parametric joint (Dempster–Laird–Rubin; Schafer 1997).** Write down one joint distribution (e.g. multivariate normal) and impute from its conditionals. Coherent, but hostage to the joint being right — and a single convenient family rarely fits mixed-scale tabular data.

The ladder starts at the cheapest honest answer (the column mean) and climbs by progressively using more of the row: local neighbours, then chained per-column regressions, then non-linear per-column predictors.

## The fixed substrate

The whole evaluation loop is frozen and must not be touched. For each dataset it loads the raw matrix, **standardizes every feature to zero mean / unit variance on the full data** (so the matrix entering the imputer is already z-scored — RMSE is measured in standard-deviation units), then corrupts it with **20% MCAR** (each cell dropped independently with probability 0.20, never an entire row or column). The corrupted matrix `X_missing` (with `NaN`s) is handed to the imputer; the completed matrix is scored two ways and the loop fills any residual `NaN` with column means as a guard. The imputer never sees the labels `y` and never sees the true values at the masked cells.

## The editable interface

Exactly one region is editable — the `CustomImputer` class (a scikit-learn `BaseEstimator, TransformerMixin`) in `scikit-learn/custom_imputation.py`, lines 36–142. The contract is the sklearn transformer triple:

- `fit(self, X, y=None) -> self` — learn the imputation model from `X` (a `(n_samples, n_features)` array with `NaN` for missing). Must not use `y`.
- `transform(self, X) -> X_imputed` — return a same-shape array with **no** `NaN`s (finite values).
- `fit_transform(self, X, y=None)` — the one-shot path the loop actually calls (`imputer.fit_transform(X_missing)`).

Available libraries: `numpy`, `scipy`, `scikit-learn` (all submodules — `sklearn.impute`, `sklearn.ensemble`, `sklearn.neighbors`, `sklearn.linear_model`, …). Every method on the ladder is a fill of this same contract. The starting point is the scaffold default: **unconditional mean imputation** — per-column means in `fit`, scatter them into the holes in `transform`.

```python
# EDITABLE region of custom_imputation.py — default fill (unconditional mean imputation)
class CustomImputer(BaseEstimator, TransformerMixin):
    """Custom missing data imputation algorithm.

    fit(X) -> self            : learn imputation model from X (with NaNs)
    transform(X) -> X_imputed : impute missing values in X
    """

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        # Default: compute column means for mean imputation
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        X_imputed = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X_imputed[:, j])
            X_imputed[mask, j] = self.statistics_[j]
        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# Helper available to the imputer (optional)
def compute_feature_correlations(X):
    """Pairwise correlations ignoring NaN pairs."""
    n_features = X.shape[1]
    corr = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if mask.sum() > 2:
                c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
                corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    return corr
```

## Evaluation settings

Three datasets spanning sizes, feature counts, and task types, each corrupted with **20% MCAR**:

- **Breast Cancer Wisconsin** — 569 samples, 30 features, binary classification.
- **Wine** — 178 samples, 13 features, 3-class classification.
- **California Housing** — 5,000 samples (subsampled), 8 features, regression.

Two metrics per dataset:

- **rmse** — root mean squared error between imputed and true values *on the masked entries only*, in the standardized units of the substrate (**lower is better**).
- **downstream_score** — 5-fold cross-validated accuracy (breast_cancer, wine) or R² (california) of a fixed `GradientBoosting` model trained on the completed matrix (**higher is better**).

Each baseline is run over three seeds {42, 123, 456}; the leaderboard reports per-seed and mean. The task score is the geometric mean across the three datasets of a per-dataset weighted mean of the (lower-better) rmse term and the (higher-better) downstream term.
