# AdaBoost — boosting weak learnability to strong

## The problem it solves

A *weak* learner returns, for any distribution over the data, a classifier with error only
slightly below 1/2 (error ≤ 1/2 − γ for some small γ > 0). A *strong* learner achieves any target
error ε. AdaBoost shows the two are **equivalent**: it converts black-box access to a weak learner
into a classifier of arbitrarily small error, using O((1/γ²) log(1/ε)) calls, *without knowing γ in
advance*. It does this by adaptively reweighting the training examples — concentrating each round
on the examples the running committee gets wrong — and combining the weak hypotheses by a weighted
majority vote.

## The algorithm (AdaBoost, {−1,+1} convention)

Given (x₁,y₁),…,(x_m,y_m) with y_i ∈ {−1,+1}; a weak learner; a number of rounds T.

- Initialize D₁(i) = 1/m.
- For t = 1,…,T:
  - Train the weak learner under distribution D_t; get h_t : X → {−1,+1}.
  - Measure its weighted error ε_t = Pr_{i∼D_t}[h_t(x_i) ≠ y_i] = Σ_i D_t(i)·1[h_t(x_i)≠y_i].
  - Set the vote coefficient **α_t = ½ ln((1−ε_t)/ε_t)**.
  - Update **D_{t+1}(i) = D_t(i)·exp(−α_t y_i h_t(x_i)) / Z_t**, with Z_t the normalizer.
- Output **H(x) = sign( Σ_t α_t h_t(x) )**.

The update down-weights examples h_t classifies correctly (y_i h_t = +1 ⇒ factor e^{−α_t} < 1) and
up-weights those it misses (y_i h_t = −1 ⇒ factor e^{+α_t} > 1). The coefficient α_t is the round's
half-log-odds: large for an accurate hypothesis, zero at chance, negative (flip h_t) above 1/2.

Equivalent {0,1} form: with β_t = ε_t/(1−ε_t), update w_i ← w_i·β_t^{1−|h_t(x_i)−y_i|} and output
1 if Σ_t a_t h_t(x) ≥ ½ Σ_t a_t, where a_t = ln(1/β_t) = ln((1−ε_t)/ε_t). The symmetric coefficient
α_t is half as large because the signed product y h swings over {−1,+1} rather than {0,1}.

## Training-error bound and the equivalence

**Theorem.** The training error of H is bounded by
  (1/m)·#{i : H(x_i) ≠ y_i} ≤ Π_{t=1}^T 2√(ε_t(1−ε_t)) = Π_t √(1 − 4γ_t²) ≤ exp(−2 Σ_t γ_t²),
where ε_t = 1/2 − γ_t. Hence if every round has edge γ_t ≥ γ > 0, the training error is below ε after
**T = ⌈ (1/2γ²) ln(1/ε) ⌉** rounds. Combined with a generalization bound, this makes a weak learner
into a strong one, proving weak learnability ⟺ strong learnability.

**Proof (exponential-loss / Z_t form).** Unrolling the update from D₁ = 1/m,
  D_{T+1}(i) = exp(−y_i F(x_i)) / (m Π_t Z_t),  where F(x) = Σ_t α_t h_t(x).
Since Σ_i D_{T+1}(i) = 1, we get m Π_t Z_t = Σ_i exp(−y_i F(x_i)). Because 1[u ≤ 0] ≤ e^{−u},
  training error = (1/m)Σ_i 1[y_i F(x_i) ≤ 0] ≤ (1/m)Σ_i exp(−y_i F(x_i)) = Π_t Z_t.
For h_t ∈ {−1,+1}, Z_t = (1−ε_t)e^{−α_t} + ε_t e^{+α_t}. Minimizing over α_t:
  dZ_t/dα_t = −(1−ε_t)e^{−α_t} + ε_t e^{+α_t} = 0 ⇒ e^{2α_t} = (1−ε_t)/ε_t ⇒ α_t = ½ ln((1−ε_t)/ε_t),
and at this α_t, Z_t = (1−ε_t)√(ε_t/(1−ε_t)) + ε_t√((1−ε_t)/ε_t) = 2√(ε_t(1−ε_t)). So the product of
the minimized Z_t's gives Π_t 2√(ε_t(1−ε_t)). Finally 4ε_t(1−ε_t) = 1 − 4γ_t² and √(1−4γ_t²) ≤
e^{−2γ_t²} (from 1−u ≤ e^{−u}), giving exp(−2Σγ_t²). ∎

**Equivalent direct proof ({0,1} form, optimizing β_t).** Convexity β^x ≤ 1−(1−β)x on [0,1] gives
Σ_i w^{t+1}_i ≤ (Σ_i w^t_i)[1−(1−ε_t)(1−β_t)], so Σ_i w^{T+1}_i ≤ Π_t[1−(1−ε_t)(1−β_t)]. A mistake by
H on i forces final weight w^{T+1}_i ≥ D(i)(Π_t β_t)^{1/2}, so Σ_i w^{T+1}_i ≥ ε·(Π_t β_t)^{1/2}.
Squeezing: ε ≤ Π_t [1−(1−ε_t)(1−β_t)]/√β_t. Each factor f(β)=ε_t β^{−1/2}+(1−ε_t)β^{1/2} is minimized
at β_t = ε_t/(1−ε_t), where it equals 2√(ε_t(1−ε_t)). Same bound.

## Why exponential reweighting — the loss view

AdaBoost is greedy coordinate descent on the **exponential loss** (1/m) Σ_i exp(−y_i F(x_i)), a
smooth convex surrogate that upper-bounds the 0/1 training error (since 1[yF≤0] ≤ e^{−yF}). Each
round adds one term α_t h_t to F, choosing h_t and α_t to cut the loss maximally; the optimal α_t is
the half-log-odds above, and the optimal per-round loss factor is Z_t = 2√(ε_t(1−ε_t)). The
exponential punishes confidently-wrong predictions sharply, which is the pressure that decorrelates
errors across rounds — and that keeps pressure on low-margin examples even after the training error
reaches zero.

## Generalization and the margin explanation

A direct VC analysis bounds the final hypothesis: if the base class H has VC-dimension d, the class
of T-round combinations has VC-dimension ≤ 2(d+1)(T+1)log₂(e(T+1)) — growing roughly linearly in T —
so the VC bound ε_g ≤ ε̂ + Õ(√(dT/m)) predicts overfitting for large T. Empirically, AdaBoost often
does *not* overfit: test error keeps falling well past zero training error.

The **margins explanation** resolves this. Define margin(x,y) = y·(Σ_t α_t h_t(x))/(Σ_t α_t) ∈
[−1,+1] — the weighted vote margin, positive iff H is correct, magnitude = confidence. One can prove
that for any θ > 0, with high probability,
  ε_g ≤ P̂[ margin(x,y) ≤ θ ] + O( √( d / (m θ²) ) ),
a bound **independent of T**. Continued boosting can keep reducing the low-margin tail even after
training error is zero, so P̂[margin ≤ θ] can shrink without paying an explicit T penalty — explaining
the resistance to overfitting that the raw-VC view cannot.

## Exhaustive stump implementation

```python
import numpy as np

def fit_stump(X, y, w):
    """Least weighted-error single-feature threshold stump. Returns (predict, eps)."""
    m, n = X.shape
    best = None
    for j in range(n):
        values = np.sort(np.unique(X[:, j]))
        mids = (values[:-1] + values[1:]) / 2.0
        thresholds = np.concatenate(([-np.inf], mids, [np.inf]))
        for thr in thresholds:
            for polarity in (+1, -1):
                pred = np.where(polarity * (X[:, j] - thr) >= 0, 1, -1)
                eps = float(np.sum(w * (pred != y)))     # weighted error under w
                if best is None or eps < best[0]:
                    best = (eps, j, thr, polarity)
    eps, j, thr, polarity = best
    return (lambda Z: np.where(polarity * (Z[:, j] - thr) >= 0, 1, -1)), eps

def adaboost(X, y, T):                                    # y in {-1, +1}
    m = len(y)
    w = np.full(m, 1.0 / m)                               # D_1: uniform
    hyps, alphas = [], []
    for t in range(T):
        h, eps = fit_stump(X, y, w)                       # measure edge; no gamma needed
        eps = np.clip(eps, 1e-12, 1 - 1e-12)
        alpha = 0.5 * np.log((1 - eps) / eps)             # minimizes Z_t
        w = w * np.exp(-alpha * y * h(X))                 # up-weight the misses
        w = w / w.sum()                                   # = dividing by Z_t
        hyps.append(h); alphas.append(alpha)
    return hyps, alphas

def predict(hyps, alphas, X):
    F = np.zeros(X.shape[0])
    for a, h in zip(alphas, hyps):
        F += a * h(X)                                     # F(x) = sum_t alpha_t h_t(x)
    return np.where(F >= 0, 1, -1)                        # weighted-majority vote
```
