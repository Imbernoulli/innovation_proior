## Research question

A tabular data matrix arrives with holes in it: entries are `NaN`, scattered across columns and rows. Almost nothing downstream tolerates a `NaN` — a gradient-boosted model cannot put one in a split, and a distance between two rows is undefined the moment one coordinate is absent. The matrix has to be turned into a finite-valued completion of the same shape. The single thing being designed is the **imputer itself**: how feature dependencies are exploited, how the fills are produced, and how completed values are generated from a matrix that still contains `NaN`. Everything around it — the datasets, the corruption, the scaling, the downstream model — is fixed.

## Prior art / Background / Baselines

- **Listwise deletion.** Drop every row with a hole and run on what remains.
- **Unconditional mean / median imputation.** Fill every hole in a column with a fixed central value for that column.
- **Hot-deck / neighbour donation.** Copy a value from a similar observed record into each hole.
- **Regression imputation.** Predict each incomplete column from the others using regression.
- **EM under a parametric joint.** Specify a joint distribution and impute from its conditionals.

## Fixed substrate / Code framework

The whole evaluation loop is frozen and must not be touched. For each dataset it loads the raw matrix, standardizes every feature to zero mean / unit variance on the full data, then corrupts it with **20% MCAR** (each cell dropped independently with probability 0.20, never an entire row or column). The corrupted matrix `X_missing` is handed to the imputer; the completed matrix is scored two ways and the loop fills any residual `NaN` with column means as a guard. The imputer never sees the labels `y` and never sees the true values at the masked cells.

## Editable interface

Exactly one region is editable — the `CustomImputer` class (a scikit-learn `BaseEstimator, TransformerMixin`) in `scikit-learn/custom_imputation.py`, lines 36–142. The contract is the sklearn transformer triple:

- `fit(self, X, y=None) -> self` — learn the imputation model from `X` (with `NaN`s). Must not use `y`.
- `transform(self, X) -> X_imputed` — return a same-shape array with **no** `NaN`s (finite values).
- `fit_transform(self, X, y=None)` — the one-shot path the loop actually calls (`imputer.fit_transform(X_missing)`).

Available libraries: `numpy`, `scipy`, `scikit-learn` (all submodules — `sklearn.impute`, `sklearn.ensemble`, `sklearn.neighbors`, `sklearn.linear_model`, …). The starting point is the scaffold default: **unconditional mean imputation** — per-column means in `fit`, scatter them into the holes in `transform`.

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
