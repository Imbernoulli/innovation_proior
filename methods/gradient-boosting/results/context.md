## Research question

We are given a training sample `{y_i, x_i}` and want an estimate `F(x)` of the function
`F*(x)` that minimizes the expected value of some specified loss `L(y, F(x))` over the joint
distribution of `(y, x)`. The losses we actually care about are varied: half squared error
`(y - F)^2/2` and absolute error `|y - F|` for real-valued regression; the Huber loss for
regression that has to survive heavy-tailed noise and outliers; and the negative binomial
log-likelihood `log(1 + e^{-2yF})` for `y in {-1, 1}` classification (and its multiclass
generalization). The practical setting is "data mining" data — large, mixed-type, with
missing values, with outliers and mislabeled cases — where a method must be accurate but also
robust and interpretable.

A very general and successful way to build `F` is the **additive expansion**

```
F(x) = sum_{m=1}^{M} beta_m * h(x; a_m),
```

where each `h(x; a)` is a simple parameterized base learner (a small regression tree, a
wavelet, a sigmoid unit). The trouble is fitting it. Doing the full joint optimization over
all `{beta_m, a_m}` at once is infeasible, so one fits it greedily, one term at a time, in a
forward stagewise manner. But that greedy per-stage subproblem,

```
(beta_m, a_m) = argmin_{beta, a} sum_i L(y_i, F_{m-1}(x_i) + beta * h(x_i; a)),
```

only has a clean, fast solution for a couple of special losses. For half squared error it reduces
to "fit the base learner to the current residuals" by least squares. For the exponential loss
in two-class classification it reduces to a known reweighting algorithm. For everything else —
absolute error, Huber, binomial deviance, multiclass deviance — there is no convenient
fitting algorithm for the subproblem, and the bespoke derivations that do exist (one per loss)
are each their own piece of machinery. The precise goal is a *single* procedure that fits a
forward-stagewise additive model beyond those special cases, without deriving a new optimizer
for every loss, while retaining the robustness and interpretability of regression trees.

## Background

**Forward stagewise additive modeling.** Additive expansions of the form above sit at the
heart of many approximation methods — neural networks (Rumelhart, Hinton and Williams 1986),
radial basis functions, MARS (Friedman 1991), wavelets, support vector machines. When the
joint fit is infeasible, the standard fallback is the greedy stagewise fit: at stage `m`, fix
`F_{m-1}` and add one new term, never readjusting the terms already entered. This is what
distinguishes "stagewise" from "stepwise" (which re-optimizes earlier coefficients). In signal
processing the squared-error version with an overcomplete wavelet dictionary is **matching
pursuit** (Mallat and Zhang 1993); in machine learning the classification version is
**boosting**.

**Numerical optimization as a chain of increments.** When a model `F(x; P)` is parameterized,
function optimization becomes parameter optimization `P* = argmin_P Phi(P)`, and numerical
methods express the solution as a sum of increments `P* = sum_m p_m`, where `p_0` is an
initial guess and each later `p_m` is a step computed from the preceding ones. The simplest is
**steepest descent**: compute the current gradient `g_m = [d Phi / d P]_{P = P_{m-1}}`, step
along the negative gradient `p_m = -rho_m * g_m`, and set the step length by a **line search**
`rho_m = argmin_rho Phi(P_{m-1} - rho * g_m)`. The whole solution is a running sum of such
gradient steps. This is a fully general optimizer — it needs only that `Phi` be
differentiable — but as stated it lives in *parameter* space.

**Loss functions and what they buy.** For a real-valued target, half squared error is convenient
but gives an outlier influence that grows with the square of its residual, so a few bad cases
dominate the fit. Absolute error bounds each residual's influence (its gradient is just a
sign) and is robust, but is awkward to optimize directly with a least-squares learner. The
Huber loss interpolates: quadratic for small residuals (statistical efficiency under normal
noise), linear beyond a transition point `delta` (robustness to long tails), where `delta` is
commonly set to a quantile of the residual magnitudes so that a controlled fraction of points
are treated as outliers. For two-class `y in {-1, 1}`, the natural statistical criterion is
the binomial deviance `log(1 + e^{-2yF})`, whose minimizer makes `F` the (symmetric) logit of
the class probability `F(x) = 0.5 * log[P(y=1|x) / P(y=-1|x)]`; the deviance grows only
linearly in the margin `yF` for badly misclassified points, so it is far gentler on noisy
labels than a criterion that grows exponentially in the margin.

**Regression trees as the base learner.** A regression tree (CART; Breiman, Friedman, Olshen
and Stone 1984) partitions the input space into disjoint regions `{R_j}` (its terminal nodes)
and predicts a constant in each:

```
h(x; {b_j, R_j}) = sum_{j=1}^{J} b_j * 1(x in R_j).
```

It is grown greedily by recursive binary splits that minimize squared error, and its only
size knob is the number of terminal nodes `J`. Trees handle mixed-type and missing data, are
invariant to monotone transformations of individual inputs (they use only order information),
are resistant to long-tailed *input* distributions and irrelevant inputs, and are
interpretable. Their well-known weakness is high variance — a single tree is an unstable,
inaccurate predictor — which is exactly the weakness that combining many of them addresses.
Because a tree is itself a sum of `J` indicator basis functions over disjoint regions, an
optimization over its `J` terminal values decomposes into `J` independent one-dimensional
problems, one per region.

**An ANOVA reading of tree size.** Any function decomposes into main effects, two-variable
interactions, three-variable interactions, and so on (the ANOVA decomposition). A tree with
`J` terminal nodes can represent interactions of order at most `min(J - 1, n)`: stumps
(`J = 2`) capture main effects only, `J = 3` adds pairwise interactions, etc. Many real target
functions are well approximated by low-order interactions, so small trees are often enough and
large trees are rarely necessary.

## Baselines

These are the prior methods a new general boosting procedure would be measured against and
would react to.

**Steepest descent in parameter space (numerical optimization).** As above: increments
`p_m = -rho_m g_m` with a line search for `rho_m`, summed into `P*`. Fully general for any
differentiable objective. **Gap:** it optimizes a fixed finite parameter vector. When the
model is a *nonparametric* additive expansion whose components are chosen adaptively from a
huge class (all small trees), there is no fixed `P` to take gradients of — the "directions"
available are constrained to whatever the base learner can represent, and ordinary
parameter-space steepest descent does not say how to respect that constraint.

**Forward stagewise least-squares / matching pursuit (Mallat and Zhang 1993).** Greedily add
the basis function `beta h(x; a)` that most reduces squared error to the current residual:
`(beta_m, a_m) = argmin sum_i [y_i - F_{m-1}(x_i) - beta h(x_i; a)]^2`, equivalently fit the
base learner to the residual `y_i - F_{m-1}(x_i)`. Simple and effective for regression.
**Gap:** it is welded to squared-error loss. Squared error is the wrong criterion for
classification and is non-robust for regression, and "fit the residuals" does not obviously
generalize to a loss whose stage subproblem has no least-squares form.

**AdaBoost (Freund and Schapire 1996, 1997).** For `y in {-1, 1}`, maintain weights `w_i` over
training examples (initially uniform). Each round, fit a classifier `h_m` under the current
weights, compute its weighted error `err_m`, set its coefficient
`alpha_m = 0.5 * log((1 - err_m) / err_m)`, add `alpha_m h_m` to the ensemble, and reweight
`w_i <- w_i * exp(-alpha_m * y_i * h_m(x_i))` (renormalized). Misclassified points therefore
gain a factor `exp(2 * alpha_m)` relative to correctly classified points, so each round
focuses on what is still hard; the final classifier is
`sign(sum_m alpha_m h_m(x))`. It is strikingly accurate and modular. **Gap:** it is specific
to two-class classification with one particular implicit criterion; it does not provide a
recipe for regression or for general differentiable losses; and its implicit loss penalizes
large-margin mistakes very steeply, which makes it brittle on noisy labels and outliers.

**Additive logistic regression / Newton-style boosting (Friedman, Hastie and Tibshirani
2000).** This recasts AdaBoost as forward stagewise additive modeling that minimizes the
exponential criterion `E[e^{-yF}]`, whose population minimizer is the symmetric logit
`0.5 log[P(y=1|x)/P(y=-1|x)]`; the AdaBoost reweighting falls out as Newton-like updates on
that criterion. It then notes the exponential criterion and the binomial deviance agree to
second order around `F = 0` but the deviance is gentler in the tails, and derives a
likelihood-based procedure (LogitBoost) by taking Newton steps on the binomial log-likelihood:
form the working response `z_i = (y*_i - p_i) / (p_i (1 - p_i))` with weights `p_i(1 - p_i)`
and fit by weighted least squares; for a tree, the optimal terminal-node constant is the
corresponding per-leaf Newton step. **Gap:** each loss still gets its own bespoke derivation
(exponential -> AdaBoost-style, binomial -> LogitBoost-style), one Newton machinery per
criterion rather than a single mechanism for an arbitrary differentiable loss; and the Newton
step divides by an estimated second derivative `p(1 - p)`, which vanishes as `p -> 0` or
`p -> 1`, making the updates numerically unstable exactly where confident points pile up.

**A single bagged or boosted tree as a predictor.** A single CART tree is interpretable but
high-variance and inaccurate. Bagging averages many trees fit to bootstrap samples and reduces
variance, but it is purely a variance-reduction device: with low-variance, high-bias base
learners (e.g. stumps) it barely helps, since it cannot reduce bias. **Gap:** averaging
independent trees does not sequentially attack the residual error a loss function still
sees — it does not descend a chosen loss.

## Evaluation settings

The natural yardsticks at the time:

- **Regression** on real-valued targets, measured by an approximation-inaccuracy metric such
  as scaled absolute error of the estimate relative to the optimal constant, and by RMSE on a
  held-out set. Robustness is probed by injecting different error distributions (normal vs.
  very heavy-tailed "slash" noise) at a controlled signal-to-noise ratio.
- **Binary and multiclass classification**, measured by misclassification error rate and by
  the loss itself (deviance / negative log-likelihood); probabilities are read off the fitted
  logit.
- **Monte-Carlo simulation over many randomly generated target functions** of fixed input
  dimension (e.g. `n = 10`), so that performance is summarized over a distribution of targets
  rather than one example; tree size `J` and the number of terms `M` are swept.
- The standard base learner is a small regression tree with a fixed number of terminal nodes
  `J`, grown best-first; model complexity is governed by `M` (number of trees), `J` (tree
  size), and a shrinkage knob. The number of terms and shrinkage are selected on a left-out
  test sample or by cross-validation. For the concrete downstream pipeline, the base learner
  is a depth-3 decision tree, the number of rounds is in the low hundreds, the learning rate
  is around 0.1, and metrics are test accuracy (classification) and test RMSE (regression).

## Code framework

The procedure plugs into a generic forward-stagewise harness over a CART regression-tree
learner that already exists. The available pieces are a tree learner fit by least squares
(`DecisionTreeRegressor`), a way to initialize the ensemble's constant prediction, and an outer
loop that adds one tree-shaped correction at a time. The unresolved pieces are the per-round
quantity handed to the tree, the terminal-node contribution, and the accumulation rule.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class BoostingMachine:
    """Forward-stagewise additive model over CART regression trees.

    F(x) = init_prediction + sum_m (contribution of tree m).
    The per-round target, the per-leaf value, and the accumulation rule are left open.
    """

    def __init__(self, loss, n_rounds=100, max_depth=3, learning_rate=0.1):
        self.loss = loss                  # a differentiable loss L(y, F)
        self.n_rounds = n_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.init_prediction = 0.0

    def _init_prediction(self, y):
        # The best constant model under the loss (the ensemble's starting point).
        # TODO: the constant we start the expansion from.
        raise NotImplementedError

    def _per_round_target(self, y, F):
        # What the next tree should be fit to, given the current ensemble values F.
        # TODO: the object we will define here.
        raise NotImplementedError

    def _leaf_value(self, y, F, leaf_sample_indices):
        # The constant this leaf should contribute, given the current F.
        # TODO: how far to step within this region.
        raise NotImplementedError

    def fit(self, X, y):
        F = np.full(len(y), self._init_prediction(y))
        self.init_prediction = F[0]
        for m in range(self.n_rounds):
            target = self._per_round_target(y, F)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, target)                       # least-squares fit
            leaves = tree.apply(X)
            for leaf in np.unique(leaves):
                idx = np.where(leaves == leaf)[0]
                # TODO: set this leaf's contribution and fold it into F
                pass
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        F = np.full(X.shape[0], self.init_prediction)
        # TODO: accumulate the trees' contributions into F
        raise NotImplementedError
```
