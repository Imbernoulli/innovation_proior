# AdaBoost: Adaptive Weak-to-Strong Boosting

## Method

Given training data `(x_1,y_1),...,(x_m,y_m)` with `y_i in {-1,+1}`, a weak learner, and rounds `T`:

1. Initialize `D_1(i)=1/m`.
2. For `t=1,...,T`:
   - Train the weak learner using distribution `D_t`; get `h_t:X->{-1,+1}`.
   - Measure weighted error
     `epsilon_t = sum_i D_t(i) 1[h_t(x_i) != y_i]`.
   - If `epsilon_t=0`, stop with this perfect weighted-sample classifier as a terminal limiting case. If `epsilon_t>=1/2` in the binary setting, flip the hypothesis when the weak-learner class permits it, or reject/stop because the round has no positive edge.
   - Set the confidence weight
     `alpha_t = (1/2) log((1-epsilon_t)/epsilon_t)`.
   - Update
     `D_{t+1}(i) = D_t(i) exp(-alpha_t y_i h_t(x_i)) / Z_t`.
3. Output
   `H(x) = sign(sum_t alpha_t h_t(x))`.

Correctly classified examples are multiplied by `exp(-alpha_t)` and incorrectly classified examples by `exp(alpha_t)`, so later weak learners focus on current errors. A lower positive `epsilon_t` gives a larger `alpha_t`, so better weak hypotheses also count more in the final vote.

Equivalent `{0,1}` primary-source form:

`beta_t = epsilon_t/(1-epsilon_t)`,

`w_i^{t+1} = w_i^t beta_t^(1-|h_t(x_i)-y_i|)`,

and the final vote uses coefficient `log(1/beta_t)`.

## Guarantee

The training error satisfies

`(1/m) |{i:H(x_i)!=y_i}| <= product_t 2 sqrt(epsilon_t(1-epsilon_t))`.

Writing `epsilon_t = 1/2 - gamma_t`,

`(1/m) |{i:H(x_i)!=y_i}| <= product_t sqrt(1-4 gamma_t^2) <= exp(-2 sum_t gamma_t^2)`.

Therefore, if every round has edge `gamma_t >= gamma > 0`, then `T >= (1/(2 gamma^2)) log(1/epsilon)` rounds drive training error below `epsilon`. The weak learner's edge does not need to be known in advance; each round's observed error sets both the update strength and the vote weight.

## Proof Sketch

For the `{0,1}` update, convexity gives

`sum_i w_i^{t+1} <= (sum_i w_i^t) [1-(1-epsilon_t)(1-beta_t)]`.

A final-vote mistake implies the mistaken example keeps at least `D(i)(product_t beta_t)^(1/2)` final weight. Combining the upper and lower bounds on total final weight gives

`error <= product_t ([1-(1-epsilon_t)(1-beta_t)] / sqrt(beta_t))`.

Minimizing each factor over `beta_t` yields `beta_t=epsilon_t/(1-epsilon_t)`, and the minimized factor is `2 sqrt(epsilon_t(1-epsilon_t))`.

## Executable Form

```python
import numpy as np


def adaboost(X, y, rounds, weak_learner_factory):
    # y must be in {-1, +1}; weak learner must support sample_weight.
    m = len(y)
    w = np.full(m, 1.0 / m)
    hypotheses = []
    alphas = []

    for _ in range(rounds):
        h = weak_learner_factory().fit(X, y, sample_weight=w)
        pred = h.predict(X)
        err = float(np.sum(w * (pred != y)))

        if err <= 0.0:
            hypotheses.append(h)
            alphas.append(np.inf)
            break
        if err >= 0.5:
            # No positive binary edge: flip externally if valid, otherwise reject/stop.
            break

        alpha = 0.5 * np.log((1.0 - err) / err)
        w *= np.exp(-alpha * y * pred)
        w /= np.sum(w)

        hypotheses.append(h)
        alphas.append(alpha)

    return hypotheses, np.array(alphas)


def predict(hypotheses, alphas, X):
    score = np.zeros(X.shape[0])
    for alpha, h in zip(alphas, hypotheses):
        if np.isinf(alpha):
            return h.predict(X)
        score += alpha * h.predict(X)
    return np.where(score >= 0, 1, -1)
```

The fetched canonical library artifact is `code/sklearn_weight_boosting.py`. Its current `AdaBoostClassifier` implements discrete SAMME: it fits with `sample_weight`, measures weighted `estimator_error`, stops on perfect classification, rejects errors at least `1 - 1/n_classes`, uses `log((1-error)/error) + log(n_classes-1)` as the estimator weight, and increases only incorrectly classified positive-weight samples. In the binary case the extra `log(n_classes-1)` term is zero, and this is algebraically equivalent to the signed update above up to the conventional factor `estimator_weight = 2 alpha_t` and a per-round normalization.
