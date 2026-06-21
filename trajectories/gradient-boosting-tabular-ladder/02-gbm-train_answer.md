AdaBoost's adaptive reweighting works, but it is wired to a single thing: a $0/1$ "right or wrong" indicator on a classification label. Each round it crushes the weight of correctly-classified examples and lifts the misclassified ones — a clever way to descend *one particular* loss, the exponential loss of the margin, but the only loss it knows. The committee only ever hears "this example is wrong," never "this example's prediction is off by $+3.2$ and the loss there is steep." If I want to predict a continuous target, fit a Huber loss robust to outliers, or minimize a Poisson loss for counts, the reweighting trick has no handle. I want to keep the boosting structure — an additive model built one tree at a time, each tree attacking what the committee gets wrong — but replace the classification-specific reweighting with something that works for *any* differentiable loss.

I propose the **Gradient Boosting Machine**, and the idea that unlocks it is to recast boosting as steepest descent in *function space*. The model is an additive expansion $F(x)=\sum_m h_m(x)$, built greedily: having $F_{m-1}$, I add one more term to reduce the total loss $\sum_i L(y_i, F(x_i))$. The exact greedy step $h_m=\arg\min_h\sum_i L(y_i, F_{m-1}(x_i)+h(x_i))$ is an intractable joint minimization over a function for a general loss, so I need an approximation — and here is the reframing. Forget for a moment that $F$ is parameterized by trees and think of the *prediction vector* $\big(F(x_1),\dots,F(x_n)\big)$ as a point in $\mathbb{R}^n$. The total loss $\Phi=\sum_i L(y_i, F(x_i))$ is then just a function on $\mathbb{R}^n$, and its direction of steepest descent is the negative gradient, whose $i$-th component is

$$g^i_m = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}.$$

If I could move each prediction independently by a small step along $-\nabla\Phi$, I would reduce the loss — ordinary gradient descent. But I cannot move the $F(x_i)$ independently; I am constrained to move along directions a *tree* can produce, because the only thing I am allowed to add is a tree $h_m(x)$. So the negative-gradient vector lives in the unconstrained $\mathbb{R}^n$ and I must *project* it onto the space of functions my base learner can represent.

That projection is the whole method. The vector $(g^1_m,\dots,g^n_m)$ is the ideal per-example nudge — the **pseudo-residual**, the amount each prediction "wants" to move to lower the loss fastest — and I fit a regression tree to *them* by least squares,

$$h_m = \arg\min_h \sum_i \big(g^i_m - h(x_i)\big)^2,$$

treating the negative gradient as a plain regression target. That tree is the closest realizable approximation, in the base-learner's class, to the steepest-descent direction. The crucial consequence is that the loss has vanished from the base learner's job entirely: the tree always solves a plain squared-error regression, no matter what $L$ is, and *all* the loss-specific information is concentrated in one place — the formula for $g^i_m$. Swap the loss, swap one gradient expression, and the entire machine retargets. That is exactly the generality AdaBoost lacked, and AdaBoost is recovered as the special case $L(y,F)=\exp(-yF)$: its negative gradient $y_i\exp(-y_iF_{m-1}(x_i))$ is large exactly on the examples the current margin gets wrong, scaled by the weight $\exp(-y_iF_{m-1}(x_i))$ that *is* the AdaBoost reweighting.

Two refinements make the raw idea work. First, fitting the tree to the pseudo-residuals fixes the tree's *partition* of the input space — its leaves $R_{jm}$ — but least squares would set each leaf's output to the mean pseudo-residual, which is only optimal when $L$ is squared error. For a general loss I do better by choosing each leaf value by an exact line search *in the loss itself*,

$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i\in R_{jm}} L\big(y_i, F_{m-1}(x_i)+\gamma\big),$$

a separate one-dimensional optimization per leaf — for logistic deviance a single closed-form Newton step. So the tree's *structure* is chosen by least squares on the gradient while its *leaf values* are calibrated by the real loss: the cheap squared-error fit finds the partition, the exact line search sets the magnitudes. Second, taking the full optimal step every round overfits, so I shrink each step by a learning rate $\nu\in(0,1]$,

$$F_m(x) = F_{m-1}(x) + \nu\,\gamma_{jm}\quad\text{for } x\in R_{jm},$$

where small $\nu$ means more rounds but better generalization — the regularization knob that trades trees for accuracy. The stagewise procedure is then: compute pseudo-residuals from the current loss gradient, fit a regression tree to them, line-search each leaf, take a shrunk step.

The cost is determined by how the tree's structure is found. The structure comes from a `splitter="best"` regression tree, which for each candidate feature *pre-sorts* the examples by that feature value and scans every adjacent pair as a candidate threshold, picking the split that most reduces squared-error impurity on the pseudo-residuals. This is **exact** split finding — every possible threshold on every feature is evaluated, so the chosen split is provably optimal for the node — but the price is an inner loop that runs over all $n$ examples for all $d$ features at every node of every tree. On a dense dataset with tens of millions of rows the per-round cost is dominated by this exhaustive pre-sorted scan, and that is the wall I can already see: not *what* the trees fit, but *how* the splits are found.

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
