**Problem (from step 1).** AdaBoost's adaptive reweighting descends only one loss — the exponential
loss of the classification margin — because it is wired to a 0/1 "right/wrong" indicator. It has no
handle on a continuous target, a robust loss, or any other differentiable objective; the committee only
ever hears *that* an example is wrong, never *how wrong*.

**Key idea.** The **Gradient Boosting Machine**: recast boosting as steepest descent in function space.
View the prediction vector (F(x₁),…,F(xₙ)) as a point in ℝⁿ and the total loss Σᵢ L(yᵢ,F(xᵢ)) as a
function on it. The negative gradient gives the ideal per-example nudge — the **pseudo-residual**
rᵢ = −[∂L(yᵢ,F(xᵢ))/∂F(xᵢ)]_{F=F_{m−1}}. Fit a regression tree to the pseudo-residuals by least
squares (this *projects* the steepest-descent direction onto the base-learner class); then choose each
leaf value by an exact line search in the *true* loss, γ_{jm} = argmin_γ Σ_{xᵢ∈R_{jm}} L(yᵢ,
F_{m−1}(xᵢ)+γ); then take a shrunk step F_m = F_{m−1} + ν·γ_{jm}.

**Why it works.** All loss-specific information is concentrated in one place — the pseudo-residual
formula — so the tree always solves a plain squared-error regression and the entire machine retargets to
any differentiable loss by swapping a single gradient expression. AdaBoost falls out as the special case
L = exp(−yF) (its reweighting *is* that loss's negative gradient). The two-stage choice — least-squares
fit for the tree structure, exact line search for the leaf magnitudes — makes it accurate across losses,
and shrinkage ν regularizes. Its limit is *speed*: the tree structure is found by **exact** split
finding — pre-sort every feature, scan every candidate threshold at every node — which on tens of
millions of rows makes a single boosting iteration extremely expensive.

**Change / code.** The per-stage fit: compute pseudo-residuals (negative gradient), fit a regression
tree by exact least-squares split finding, line-search the leaves, shrink. Real code from scikit-learn
`sklearn/ensemble/_gb.py` (`BaseGradientBoosting._fit_stage`, tag 1.3.2):

```python
def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask,
               random_state, X_csc=None, X_csr=None):
    """Fit another stage of ``_n_classes`` trees to the boosting model."""
    assert sample_mask.dtype == bool
    loss = self._loss
    original_y = y
    # gradients are evaluated at iteration i-1, so use a copy of raw_predictions
    raw_predictions_copy = raw_predictions.copy()

    for k in range(loss.K):
        if loss.is_multi_class:
            y = np.array(original_y == k, dtype=np.float64)

        # pseudo-residuals: the negative gradient of the loss at the current model
        residual = loss.negative_gradient(
            y, raw_predictions_copy, k=k, sample_weight=sample_weight
        )

        # induce a regression tree on the residuals (exact, pre-sorted "best" splitter)
        tree = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter="best",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=random_state,
            ccp_alpha=self.ccp_alpha,
        )

        if self.subsample < 1.0:
            sample_weight = sample_weight * sample_mask.astype(np.float64)

        X = X_csr if X_csr is not None else X
        tree.fit(X, residual, sample_weight=sample_weight, check_input=False)

        # line-search each leaf in the true loss, then take a shrunk learning_rate step
        loss.update_terminal_regions(
            tree.tree_, X, y, residual, raw_predictions, sample_weight, sample_mask,
            learning_rate=self.learning_rate, k=k,
        )
        self.estimators_[i, k] = tree

    return raw_predictions
```

`update_terminal_regions` performs Friedman's per-leaf γ_{jm} optimization (e.g. for binomial deviance,
the closed-form Newton step Σw·r / Σw·p(1−p)) and applies the `learning_rate` shrinkage when adding the
leaf values to the running prediction. The `splitter="best"` of `DecisionTreeRegressor` is the exact,
pre-sorted, scan-all-thresholds split search — the speed bottleneck the next rung attacks.
