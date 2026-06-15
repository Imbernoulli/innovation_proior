# AdaBoost (and AdaBoost.R2), distilled

AdaBoost (Adaptive Boosting) turns a weak learner — one only slightly better than chance on every
distribution — into a strongly accurate ensemble by a sequential reweight-fit-vote loop. Each round
it fits a weak learner on a weighted training set, *measures* that learner's weighted error, and uses
that single number to do two things at once: set how much it down-weights the now-solved examples for
the next round, and set how much the learner counts in the final weighted vote. Its key novelty over
its predecessors is that it is *adaptive*: it never needs the weak learner's edge specified in advance,
and it gives stronger rounds proportionally more influence. AdaBoost.R2 carries the same machinery to
regression by bounding the per-example loss to `[0,1]` and combining by a weighted median.

## Problem it solves

Given only a black-box weak learner with error at most `1/2 - gamma` (`gamma > 0`) on any
distribution, produce a predictor with arbitrarily small error — for classification (binary or
multiclass) and, in the R2 variant, for real-valued regression — using a small, predictable number of
calls, without being told `gamma`.

## Key idea (classification)

Maintain a distribution `D_t` over training examples. Each round: fit `h_t` on `D_t`, measure its
weighted error `eps_t`, then reweight examples up where `h_t` was wrong and down where it was right,
so the next learner is forced onto current failures. Combine the learners by a confidence-weighted
vote, weighting `h_t` by a coefficient set from `eps_t`. The reweighting strength and the vote weight
are the *same* quantity — both driven by the measured `eps_t` — which is what makes it adaptive.

## Final algorithm (discrete AdaBoost, `{-1,+1}`)

```
Init: D_1(i) = 1/m for all i = 1..m.
For t = 1..T:
    h_t = weak_learn(D_t)                              # fit on weighted training set
    eps_t = sum_i D_t(i) * 1[h_t(x_i) != y_i]          # weighted error
    if eps_t >= 1/2: stop (no usable edge)
    alpha_t = (1/2) * log((1 - eps_t) / eps_t)         # vote weight, chosen by the bound
    D_{t+1}(i) = D_t(i) * exp(-alpha_t * y_i * h_t(x_i)) / Z_t   # Z_t renormalizes
Output: H(x) = sign( sum_t alpha_t * h_t(x) ).
```

Equivalent `{0,1}` form: with `beta_t = eps_t/(1 - eps_t)`, update
`w_i <- w_i * beta_t^(1 - |h_t(x_i) - y_i|)`, vote weight `log(1/beta_t) = 2 * alpha_t`, predict `1`
iff `sum_t log(1/beta_t) h_t(x) >= (1/2) sum_t log(1/beta_t)`. A shrinkage / learning-rate `nu` scales
the stored coefficient to trade step size for round count.

## Why those constants

- `alpha_t = (1/2) log((1 - eps_t)/eps_t)` and `beta_t = eps_t/(1 - eps_t)` are not chosen by hand;
  they *minimize the per-round factor of the training-error bound* (below). Small `eps_t` -> small
  `beta_t` -> sharp down-weighting and a large vote; `eps_t` near `1/2` -> `beta_t` near 1 -> almost
  no reweighting and almost no vote.
- The `1/2` in `alpha_t` is the `{-1,+1}` vs `{0,1}` convention: the signed margin spans width 2, the
  indicator width 1, so the signed coefficient is half the `{0,1}` vote weight.
- Multiplicative (exponential) reweighting comes from the convexity / multiplicative-weights argument:
  it makes correct examples decay geometrically and the total-weight bound a clean product.
- Stop at `eps_t >= 1/2` (binary; `>= 1 - 1/K` for SAMME multiclass): `beta_t >= 1` would flip the
  update; a learner no better than chance must not be trusted.
- Weak learner = shallow tree (stump / depth 2-3): only needs an edge, and must stay *weak* so each
  round contributes a little and later rounds still have an edge to find.

## Training-error theorem and proof

**Theorem.** `training_error(H) <= prod_t 2 sqrt(eps_t(1 - eps_t)) <= exp(-2 sum_t gamma_t^2)`, where
`eps_t = 1/2 - gamma_t`. Hence if every round has edge `>= gamma`, then `T >= (1/(2 gamma^2)) ln(1/epsilon)`
rounds give training error `< epsilon`.

**Proof (`{0,1}` weights, `W_t = sum_i w_i^t`, `W_1 = 1`).**
1. *Upper bound on surviving weight.* `beta^x <= 1 - (1-beta)x` on `x in [0,1]` (convexity). With
   `x = 1 - |h_t - y_i|`, summing against `w_i^t`: `W_{t+1} <= W_t [1 - (1-beta_t)(1-eps_t)]`, so
   `W_{T+1} <= prod_t [1 - (1-beta_t)(1-eps_t)]`.
2. *Lower bound.* A final-vote mistake on `i` forces `prod_t beta_t^(-|h_t-y_i|) >= (prod_t beta_t)^(-1/2)`;
   since `w_i^{T+1} = D_1(i) (prod_t beta_t) prod_t beta_t^(-|h_t-y_i|)`, a mistaken `i` has
   `w_i^{T+1} >= D_1(i)(prod_t beta_t)^(1/2)`. Summing over mistakes: `W_{T+1} >= error (prod_t beta_t)^(1/2)`.
3. *Squeeze.* `error <= prod_t [eps_t + (1-eps_t)beta_t] / sqrt(beta_t)`. Minimizing
   `f(beta) = eps beta^{-1/2} + (1-eps) beta^{1/2}` (set `f'=0`: `(1-eps)beta = eps`) gives
   `beta_t = eps_t/(1-eps_t)` and `f(beta_t) = 2 sqrt(eps_t(1-eps_t))`.
4. With `eps_t = 1/2 - gamma_t`: `4 eps_t(1-eps_t) = 1 - 4 gamma_t^2`, and `sqrt(1-4gamma_t^2) <= exp(-2gamma_t^2)`
   (from `1 - u <= e^{-u}`). Product gives `exp(-2 sum_t gamma_t^2)`. QED.

## Generalization

A `T`-round weighted vote over a base class of VC dimension `d` has VC dimension
`<= 2(d+1)(T+1) log_2(e(T+1))`, so the crude capacity bound grows with `T`. A sharper diagnostic is the
**margin** `y_i F(x_i)/sum_t |alpha_t| in [-1,+1]`: the sign is correctness and the magnitude is vote
confidence, so margin-based bounds track the margin distribution and base-class complexity rather than
only the round count. (The exponential loss `(1/m) sum_i exp(-y_i F(x_i))`, greedily minimized
coordinate-wise by `alpha_t`, is a useful later lens on the same update, but the guarantee above comes
from the weight argument, not from starting with that loss.)

## AdaBoost.R2 (regression)

Bound the per-example loss to `[0,1]` so the classification machinery transfers. With predictions
`y'_i`, `D = max_i |y'_i - y_i|`, per-example loss `L_i = |y'_i - y_i|/D` (linear; or `(|.|/D)^2`, or
`1 - exp(-|.|/D)`), weighted average loss `Lbar = sum_i p_i L_i`:

```
if Lbar >= 1/2: stop
beta = Lbar / (1 - Lbar)                               # in [0,1)
w_i <- w_i * beta^(1 - L_i)                            # well-predicted shrink, hard kept
vote weight c_t = log(1/beta)
```

Combine by **weighted median**: sort each input's predictions `y_t(x)`, output the one at which the
cumulative confidence `c_t` first reaches half the total — robust to a single bad learner, unlike a
mean. Default base learner `DecisionTreeRegressor(max_depth=3)`, `loss="linear"`.

## Working code

The code mirrors scikit-learn's `_weight_boosting` control flow: classifier rounds fit with
`sample_weight`, regression rounds use a weighted bootstrap, and regression prediction is a weighted
median.

```python
import numpy as np


class AdaBoost:
    """Discrete classification boosting plus bounded-loss regression boosting."""

    def __init__(self, make_weak_learner, task_type="classification",
                 n_rounds=200, learning_rate=1.0, loss="linear", random_state=None):
        self.make_weak_learner = make_weak_learner
        self.task_type = task_type
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.learners_, self.estimator_weights_, self.estimator_errors_ = [], [], []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        w = np.ones(n, dtype=float) / n                    # D_1: uniform
        if self.task_type == "classification":
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)

        for t in range(self.n_rounds):
            learner = self.make_weak_learner()

            if self.task_type == "regression":
                p = w / w.sum()
                idx = rng.choice(np.arange(n), size=n, replace=True, p=p)
                learner.fit(X[idx], y[idx])
            else:
                learner.fit(X, y, sample_weight=w)

            pred = learner.predict(X)
            p = w / w.sum()

            if self.task_type == "classification":
                incorrect = (pred != y)
                eps = float(np.average(incorrect, weights=p))
                if eps <= 0:
                    self.learners_.append(learner)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(0.0)
                    break
                if eps >= 1.0 - 1.0 / n_classes:
                    break
                learner_weight = self.learning_rate * (
                    np.log((1.0 - eps) / eps) + np.log(n_classes - 1.0)
                )
                if t != self.n_rounds - 1:
                    w = np.exp(np.log(w) + learner_weight * incorrect * (w > 0))
            else:
                mask = w > 0
                loss_vec = np.abs(pred[mask] - y[mask])
                loss_max = loss_vec.max()
                if loss_max != 0:
                    loss_vec = loss_vec / loss_max         # linear loss in [0,1]
                if self.loss == "square":
                    loss_vec = loss_vec ** 2
                elif self.loss == "exponential":
                    loss_vec = 1.0 - np.exp(-loss_vec)
                eps = float(np.dot(p[mask], loss_vec))
                if eps <= 0:
                    self.learners_.append(learner)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(0.0)
                    break
                if eps >= 0.5:
                    break
                beta = eps / (1.0 - eps)
                learner_weight = self.learning_rate * np.log(1.0 / beta)
                if t != self.n_rounds - 1:
                    w[mask] *= np.power(beta, (1.0 - loss_vec) * self.learning_rate)

            w = w / w.sum()                               # renormalize
            self.learners_.append(learner)
            self.estimator_weights_.append(float(learner_weight))
            self.estimator_errors_.append(float(eps))
        return self

    def predict(self, X):
        weights = np.asarray(self.estimator_weights_, dtype=float)
        if self.task_type == "classification":
            votes = np.zeros((len(X), len(self.classes_)))
            for weight, learner in zip(weights, self.learners_):
                pred = learner.predict(X)
                for j, cls in enumerate(self.classes_):
                    votes[:, j] += weight * (pred == cls)
            return self.classes_[np.argmax(votes, axis=1)]

        preds = np.array([lr.predict(X) for lr in self.learners_]).T   # (n, T)
        out = np.empty(preds.shape[0])
        for k in range(preds.shape[0]):
            order = np.argsort(preds[k])
            cdf = np.cumsum(weights[order])
            j = np.searchsorted(cdf, 0.5 * cdf[-1])
            out[k] = preds[k][order[min(j, len(order) - 1)]]
        return out
```
