# Context: online and stochastic (sub)gradient learning on high-dimensional sparse data (circa 2009–2010)

## Research question

We want to learn a weight vector `x ∈ R^d` from a stream of convex losses `f_1, f_2, …`,
where `d` is enormous (text/NLP feature spaces, click models, ranking) but each instance
touches only a handful of coordinates — the gradients `g_t` are extremely sparse. Stochastic
and online (sub)gradient methods all run the update `x_{t+1} = x_t − η_t g_t` (with a
projection), and `η_t` is a single global scalar, typically `η/√t`, shared across every
coordinate. In sparse high-dimensional data the coordinates are heterogeneous: a few features
fire on almost every example, while many are rare. The rare ones are often the most
informative — practitioners already pre-emphasize them by hand with schemes like TF-IDF
(Salton & Buckley 1988). The question is how to set the learning rate in an online
(sub)gradient method on such data, while retaining a provable sub-linear regret bound.

## Background

The setting is online convex optimization with the regret model. At each round the learner
predicts `x_t ∈ X ⊆ R^d`, then sees a convex loss `f_t` and a (sub)gradient `g_t ∈ ∂f_t(x_t)`.
Performance is measured by **regret** against the best fixed predictor in hindsight,
`R(T) = Σ_{t=1}^T f_t(x_t) − inf_{x∈X} Σ_{t=1}^T f_t(x)`; the goal is `R(T) = o(T)`. We often
work with **composite** losses `f_t = ℓ_t + φ`, where `φ` is a fixed regularizer (e.g. an
`ℓ_1` penalty for sparsity, or an indicator of a constraint set). Online and stochastic
convex optimization are essentially interchangeable (Cesa-Bianchi et al. 2004): a regret bound
converts to an expected-suboptimality bound for the stochastic problem by online-to-batch.

The load-bearing machinery is the **proximal / mirror-descent** view of these updates. Rather
than the bare gradient step, write the move as a trade-off between following the current
gradient and staying close to the current point in a chosen geometry:
`x_{t+1} = argmin_x { η⟨g_t,x⟩ + ηφ(x) + B_ψ(x,x_t) }`, where `B_ψ` is the Bregman divergence
of a strongly convex `ψ`. Taking `ψ(x)=½‖x‖₂²` recovers ordinary projected gradient descent;
taking `ψ` to be negative entropy gives the exponentiated-gradient / multiplicative-weights
geometry on the simplex. The companion "lazy"/dual-averaging family (Nesterov 2009; Xiao's
2010 regularized dual averaging, RDA; and the follow-the-regularized-leader algorithms of
Kalai & Vempala 2003, Hazan et al. 2006) makes the prediction from the *running average*
gradient against a strongly convex `ψ`. For all of these the regret has the same shape,
`R(T) ≤ (1/η)·B_ψ(x*,x_1) + (η/2) Σ_t ‖g_t‖²_{ψ*}`, where `‖·‖_{ψ*}` is the dual norm of the
norm under which `ψ` is strongly convex. Choosing `η ∝ 1/√T` (or, with bounded Bregman
diameter, `η_t ∝ 1/√t`, equivalently `ψ_t = √t·ψ`) yields `O(√T)` regret.

Two facts about this landscape are standard. First, the `ψ` (the geometry) is
chosen by hand and held fixed for the whole run — at most it is scaled by a time-dependent
scalar `√t`. The bound then depends on the single norm `‖·‖` picked up front and on the
gradient magnitudes in its dual norm. Second, for the Euclidean choice the resulting
`O(√T)` rate, `≈ √2·D₂·√(Σ_t‖g_t‖₂²)` with `D₂` the `ℓ₂` diameter, is **minimax optimal** in
the worst case (Abernethy et al. 2008).

The motivating empirical regime has measurable structure. In high-dimensional
text/ranking problems the data and hence the gradients are sparse and **heavy-tailed**: if
feature `i` appears with probability `p_i = min{1, c·i^{-α}}`, then over `T` rounds it fires
only about `p_i T` times, so the per-coordinate sum `Σ_i √(p_i)` is `O(log d)` for `α ≥ 2` and
`O(d^{1−α/2})` for `α ∈ (1,2)` — smaller than the `√d` that an isotropic Euclidean bound
charges.

## Baselines

- **Online / greedy projected (sub)gradient descent (Zinkevich 2003).** The canonical online
  convex optimization algorithm: `x_{t+1} = Π_X(x_t − η_t g_t)` with `η_t = 1/√t`. Its regret
  is bounded by telescoping the Euclidean contraction: `Σ_t (x_t−x*)·g_t ≤ Σ_t [‖x_t−x*‖² −
  ‖x_{t+1}−x*‖²]/(2η_t) + (η_t/2)Σ_t‖g_t‖²`; after summing the telescoping term and choosing the
  scalar step to balance distance and gradient mass, this gives
  `R(T) = O(D₂√(Σ_t‖g_t‖₂²))`. It uses the fixed Euclidean metric and a single global scalar
  step, identical for every coordinate; it needs no smoothness or strong convexity, and is
  minimax-optimal.

- **Online mirror descent / Bregman-proximal methods (Nemirovski–Yudin; Beck & Teboulle 2003;
  Bartlett et al. 2007).** Generalize the Euclidean step to an arbitrary
  strongly convex `ψ`, matching the geometry to the problem — e.g. negative entropy
  on the simplex gives the multiplicative-weights `√(log d)` dependence instead of `√d`. The
  same `(1/η)B_ψ(x*,x_1) + (η/2)Σ‖g_t‖²_{ψ*}` bound holds. `ψ` is picked by hand and a priori,
  and stays fixed for the run.

- **Regularized dual averaging / FTRL (Nesterov 2009; Xiao 2010; Kalai & Vempala 2003; Hazan et
  al. 2006).** Predict from the running average gradient against a strongly convex `ψ` scaled by
  `√t`: `x_{t+1} = argmin_x{ η⟨ḡ_{1:t},x⟩ + ηφ(x) + (1/t)ψ_t(x) }`, regret `≤ √T·ψ(x*) +
  (1/2√T)Σ‖g_t‖²_{*}`. It handles composite `φ` (genuine sparsity from `ℓ_1`) and needs no fixed
  horizon. The time scaling of `ψ` is a scalar `√t` over a fixed, isotropic metric.

- **Variable-metric / quasi-Newton subgradient methods (Shor 1972 space-dilation; BFGS,
  Fletcher 1970; Nedić 2002; Bordes et al. 2009).** Adapt a metric to local curvature. They
  assume a differentiable, often smooth objective with positive-definite Hessian bounded away
  from zero (Bordes), are deterministic, and maintain a full metric matrix.

- **Confidence-weighted / AROW (Crammer et al. 2008, 2009); second-order Perceptron
  (Cesa-Bianchi et al. 2005).** Maintain a mean `μ_t` and a covariance `Σ_t` over the weights
  and shrink the variance in the directions of observed features — second-order, per-feature
  adaptive in spirit. They come with *mistake* bounds tied to the specific run of the algorithm.

## Evaluation settings

The natural yardsticks are high-dimensional sparse classification and ranking. Text
categorization corpora (e.g. Reuters RCV1, 20 Newsgroups), large-scale named-entity / sequence
tagging with sparse lexical features, and click/ranking datasets with bag-of-features
representations — all with `d` from `10^4` to `10^6+` and per-example sparsity of a few hundred
nonzeros. Losses are the hinge loss `[1 − y_t⟨z_t,x⟩]_+` and logistic loss on `0/1`-valued
features, optionally with an `ℓ_1` or `ℓ_2` regularizer `φ`. The metric is regret (online) or,
equivalently via online-to-batch, test error / generalization after a stream; the per-iteration
cost must stay near-linear in the support of `g_t` to be usable at this scale. The baselines to
run against are the methods above (projected subgradient with `η/√t`, mirror descent, RDA/FTRL),
under matched per-step cost.

## Code framework

A generic online-learning loop over a stream has the *geometry / preconditioner* as the one
empty slot. The surrounding pieces are the projection, the subgradient oracle, and the regret
accounting.

```python
import numpy as np

class Preconditioner:
    """The geometry of the update: a strongly convex ψ defining the proximal/
    mirror-descent move. Existing choices pick a fixed metric by hand (Euclidean,
    or an entropic norm)."""
    def __init__(self, d, eta):
        self.d = d
        self.eta = eta
        # TODO: state for the geometry goes here

    def step(self, x, g):
        # TODO: return the (sub)gradient move under this geometry (e.g. its prox form)
        pass

def project(x, X):
    """Euclidean projection onto the feasible set X (e.g. an L-inf box)."""
    pass  # standard projection

def subgradient(f_t, x):
    """A subgradient of the convex loss f_t at x (e.g. hinge / logistic)."""
    pass

def online_learn(stream, X, d, eta):
    x = np.zeros(d)
    precond = Preconditioner(d, eta)
    regret_terms = []
    for f_t in stream:
        g = subgradient(f_t, x)          # g_t in ∂f_t(x_t)
        x = precond.step(x, g)           # (sub)gradient move under the chosen geometry
        x = project(x, X)                # keep x feasible
        regret_terms.append(f_t(x))
    return x
```

The open slot is entirely inside `Preconditioner`: the geometry under which the move is taken.
