A standardized $(n, d)$ matrix arrives with one fifth of its cells punched out at random — each entry independently `NaN` with probability $0.20$, missing completely at random because the substrate drops cells by a coin flip that never looks at any value. Nothing downstream tolerates a hole: the `GradientBoosting` model cannot put a `NaN` in a split, and a distance between two rows is undefined the moment one coordinate is absent. So before anything clever I owe the loop a single map, `transform`, from a table with holes to a same-shape table of finite numbers, learned once in `fit` and replayable on whatever matrix arrives. Discarding the incomplete rows is not an option masquerading as one: with $d$ columns each missing at rate $p$, the chance a row survives intact is $(1-p)^d$, which at $p=0.2$ is about $0.17$ at $d=8$, $0.055$ at $d=13$, and $0.0012$ at $d=30$ — thirty columns at twenty percent missing throws away 998 rows in a thousand, and a held-out row with a hole still cannot be completed. I have to fill, and the only material I have is the entries that are present, since the masked truth is hidden from me.

I propose to start at the honest floor: **unconditional mean imputation** — write into every hole in column $j$ that column's observed mean, the same number into every missing row of that column regardless of what the row is doing elsewhere. The choice of constant is not arbitrary, and two derivations land on the same number, which is what convinces me it is the right floor rather than merely the cheapest. The first asks which single constant $c$ is closest in squared error to the column's observed values: minimizing $\sum_{\text{obs}} (x_{ij} - c)^2$ over $c$ gives $-2\sum_{\text{obs}}(x_{ij}-c)=0$, so $c = \bar{x}_j$, the observed mean. The mean is therefore the least-squares constant predictor of the column — the best you can do at guessing an entry if you refuse to look at any other variable, which is exactly the situation a hole puts me in if I use no covariates. And because RMSE is a squared-error loss on the masked cells, and under MCAR the observed entries are an unbiased sample drawn from the same distribution as the hidden ones, the constant that minimizes squared error to the observed column is also the squared-error-optimal *no-covariate* fill for the masked cells I am scored on. The second derivation pins down the same value from a different angle: among all constants, $\bar{x}_j$ is the unique one that leaves the completed column's mean intact. Filling $n - n_{\text{obs}}$ holes with $c$ gives a completed mean $(n_{\text{obs}}\bar{x}_j + (n-n_{\text{obs}})c)/n$, which equals $\bar{x}_j$ exactly when $c = \bar{x}_j$; any other constant drags the completed mean toward $c$.

The mechanism is then one pass to compute $d$ means and one pass to scatter them. In `fit` I store `np.nanmean(X, axis=0)`, the per-column mean over the non-`NaN` rows; in `transform` I copy the input and, column by column, write each stored mean into that column's holes. This satisfies the freeze-and-replay contract for free — the means learned in `fit` fill any later table identically — and it never touches the labels. The two carried hyperparameters `random_state` and `max_iter` do nothing here; they exist only so the interface matches the richer imputers to come.

I want to be precise about what this floor forfeits, because that damage is the entire reason a ladder exists above it, and both costs are derivable. The first is variance: every filled entry equals the column mean exactly, contributing zero squared deviation, so the sum of squared deviations stays frozen at the observed part's $(n_{\text{obs}}-1)s^2_{\text{obs}}$ while the completed variance divides it by $n-1$ instead of $n_{\text{obs}}-1$, giving $s^2_{\text{completed}} = \frac{n_{\text{obs}}-1}{n-1}\,s^2_{\text{obs}}$ — strictly below the truth, and on this z-scored data where each true variance is $1$, twenty percent missing deflates it to about $0.8$. The second cost is worse for these tables: the correlations. When two columns genuinely move together, the observed pairs trace a tilted cloud, but the entries I fill in one column all sit at its mean regardless of what the partner column does in those rows, laying down a flat horizontal band with no tilt; mixing that flat band into the tilted cloud pulls the Pearson correlation toward zero. By construction the filler asserts "in these rows this variable is exactly average and tells you nothing about its partner" — the no-correlation hypothesis, smeared over a fifth of the data. So the RMSE will be punished hardest where features are most predictable from each other — I expect the worst RMSE on the wide, correlated Breast Cancer matrix, in standardized units near or above $1$ — while the downstream scores should hold up far better, because the `GradientBoosting` model re-fits on the completed matrix and can treat each per-column spike as its own region. That gap between a poor RMSE and a respectable downstream score is the budget the next rung claims, and the diagnosis it points to is already written: this is a *correlation* problem, not a scale problem, and the fix is to stop using one number per column and start reading each row's observed coordinates to fill its holes.

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    """Mean Imputation: replace missing values with column means."""

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
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


def compute_feature_correlations(X):
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
