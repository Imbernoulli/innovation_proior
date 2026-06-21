# Context: online and stochastic convex optimization with subgradients (circa 2009-2010)

## Research question

We have a stream of convex cost functions and must commit to a decision before each one
is revealed. Formally, at each round `t` the learner predicts a point `x_t` in a closed
convex set `X ⊆ R^d`, then suffers a convex loss `f_t(x_t)` and receives a subgradient
`g_t ∈ ∂f_t(x_t)`. (In the stochastic-optimization reading, `f_t` is the loss on a single
fresh example, so `g_t` is a noisy unbiased subgradient of the expected risk.) Success is
measured by **regret** against the single best fixed decision in hindsight,

```
R(T) = sum_{t=1}^{T} f_t(x_t) - inf_{x in X} sum_{t=1}^{T} f_t(x),
```

and the goal is for this to be **sublinear**, `R(T) = o(T)`, so average regret vanishes.

Many of the high-dimensional problems that motivate this work — text classification,
bag-of-words ranking — have extremely sparse input vectors: within any one example only a
few of the `d` features are non-zero, so most coordinates of `g_t` are zero on most
rounds. Rare features are often informative and discriminative, the folklore behind
inverse-document-frequency feature weightings such as TF-IDF (Salton & Buckley 1988),
where rare terms are deliberately up-weighted. The question is how to set the proximal
geometry — the step size, or more generally the preconditioner — of a subgradient method
for a stream of such gradients, with a regret guarantee, at a cost close to that of
vanilla subgradient descent so it remains usable when `d` is in the millions.

## Background

The field state is the regret-minimization view of online learning, with two closely
linked algorithmic families that are by now standard.

**The online-convex / regret frame.** Treat optimization as a repeated game (Zinkevich
2003): commit `x_t`, then an adversary reveals convex `f_t`; the yardstick is regret
against the best fixed `x*`. The basic analytic tool is the convexity hyperplane bound —
a convex function lies above its tangent, so for any `x*`,
`f_t(x_t) - f_t(x*) ≤ g_t^T (x_t - x*)`. This converts regret into a sum of linear terms
in the algorithm's own updates, which is what makes the analysis tractable.

**Mirror descent and the proximal-point view.** A clean way to see gradient descent is as
linearize-and-stay-close: minimize the linear model of `f_t` plus a penalty that keeps
the next point near the current one,
`x_{t+1} = argmin_x { η⟨g_t, x⟩ + ½||x - x_t||²_2 }`, whose stationary condition is
exactly `x_{t+1} = x_t - η g_t`. Replacing the squared Euclidean penalty by a general
**Bregman divergence** `B_ψ(x, y) = ψ(x) - ψ(y) - ⟨∇ψ(y), x - y⟩` of a strongly convex
`ψ` gives mirror descent (Nemirovski & Yudin 1983; Beck & Teboulle 2003),

```
x_{t+1} = argmin_x { η⟨g_t, x⟩ + η φ(x) + B_ψ(x, x_t) },
```

where `φ` is an optional fixed composite regularizer (e.g. an `ℓ1` penalty for sparsity).
With `ψ = ½||·||²_2` and `φ = 0` this is projected gradient descent again; with negative
entropy it becomes exponentiated-gradient / Hedge. The function `ψ` is called the proximal
term; the norm it induces, `||·||_ψ`, and its dual `||·||_{ψ*}` are what the analysis
turns on.

**The standing regret bounds.** For a fixed strongly-convex `ψ`, mirror-descent–type
first-order methods (Zinkevich 2003; Bartlett, Hazan & Rakhlin 2007; Beck & Teboulle
2003) satisfy a bound of the form

```
R_φ(T) ≤ (1/η) B_ψ(x*, x_1) + (η/2) sum_{t=1}^{T} ||g_t||²_{ψ*}.
```

Choosing `η ∝ 1/√T` (equivalently `ψ_t = √t · ψ`, removing the need to know the horizon)
gives `R_φ(T) = O(√T)`. The dual-averaging / follow-the-regularized-leader family —
Nesterov's primal-dual subgradient method (2009) and Xiao's regularized dual averaging
(RDA, 2010) — instead predict from the *running average* gradient
`ḡ_t = (1/t) Σ_{τ≤t} g_τ`,
`x_{t+1} = argmin_x { η⟨ḡ_t, x⟩ + η φ(x) + (1/t) ψ_t(x) }`, and keep `φ` intact
(rather than linearizing it), which yields genuinely sparse iterates under an `ℓ1`
penalty; its bound has the same shape,
`R_φ(T) ≤ √T ψ(x*) + (1/(2√T)) Σ_t ||g_t||²_*` (Xiao 2010, Theorem 3).

In every one of these bounds the data-dependent part is a sum of **dual norms of the
gradients**, `Σ_t ||g_t||²_{ψ*}`, and that dual norm is fixed by the a-priori choice of
`ψ`. Zinkevich's algorithm takes `ψ(x) = ½||x||²_2`; RDA takes `ψ_t = √t ψ` for a fixed
`ψ`; the shape of the metric is decided before any data is seen and is the same for every
coordinate. Abernethy et al. (2008) showed Zinkevich's `O(√T)` bound is minimax-tight, so
any improvement must come from extra structure in the input — the sparsity — rather than
from a sharper analysis of the same isotropic algorithm.

A line of older work adapts the metric to the data in other settings: space-dilation
methods (Shor 1972), variable-metric / quasi-Newton schemes such as BFGS (Fletcher 1970),
and Nedić's variable-metric subgradient thesis (2002) all assume a differentiable,
deterministic, unconstrained objective. The confidence-weighted / second-order-Perceptron
/ AROW line (Cesa-Bianchi et al. 2005; Crammer et al. 2008, 2009) maintains a
per-coordinate confidence (a covariance) and updates it multiplicatively, with mistake
bounds tied to a specific loss, using the inverse covariance.

## Baselines

These are the prior methods a new adaptive scheme is measured against and reacts to.

**Projected online gradient descent (Zinkevich 2003).** Step against the subgradient and
project back into the feasible set,
`x_{t+1} = Π_X(x_t - η g_t)`, with the decreasing schedule `η_t = η/√t`. With diameter
`sup_{x,y} ||x - y||_2 ≤ D_2` and the optimal hindsight `η`, the regret is
`Σ_t f_t(x_t) - inf_x Σ_t f_t(x) ≤ √2 D_2 (Σ_t ||g_t||²_2)^{1/2}`, i.e. `O(√(d T))` in the
dense worst case. A single isotropic step size `η_t` scales every coordinate.

**Mirror descent with a fixed proximal function (Nemirovski & Yudin 1983; Beck & Teboulle
2003).** The update `x_{t+1} = argmin_x { η⟨g_t, x⟩ + η φ(x) + B_ψ(x, x_t) }` with regret
`(1/η) B_ψ(x*, x_1) + (η/2) Σ_t ||g_t||²_{ψ*}`. The proximal `ψ` is chosen to match the
geometry of `X` (Euclidean for a box, entropy for the simplex), once, before any gradient
is seen, and identically for every coordinate.

**Regularized dual averaging / primal-dual subgradient (Nesterov 2009; Xiao 2010).**
Predict from the average gradient,
`x_{t+1} = argmin_x { η⟨ḡ_t, x⟩ + η φ(x) + (1/t) ψ_t(x) }`, with `ψ_t = √t ψ` for a fixed
strongly convex `ψ`; keeping `φ` un-linearized produces sparse iterates under an `ℓ1`
penalty, with regret `√T ψ(x*) + (1/(2√T)) Σ_t ||g_t||²_*`. For the `ℓ1` case it gives
the coordinate-wise soft-threshold
`x_{t+1,i} = sign(-ḡ_{t,i}) · η√t · [|ḡ_{t,i}| - λ]_+`. The proximal term `ψ_t = √t ψ` is
a single fixed `ψ` scaled by the global, time-dependent scalar `√t`; every coordinate is
annealed at the same `√t` rate.

**Confidence-weighted / AROW (Cesa-Bianchi et al. 2005; Crammer et al. 2008, 2009).**
Maintains a mean predictor `μ_t` and a covariance `Σ_t`, updating both on each margin
violation: `μ_{t+1} = μ_t + α_t Σ_t y_t z_t`, `Σ_{t+1} = Σ_t - β_t Σ_t z_t z_t^T Σ_t`,
with `β_t = 1/(⟨z_t, Σ_t z_t⟩ + λ)`. This is per-coordinate second-order information and
works well on sparse text. The guarantees are mistake bounds for a particular (hinge-type)
loss, and it preconditions by the inverse covariance.

## Evaluation settings

The natural yardsticks for this regime are single-pass online/stochastic protocols on
large sparse prediction problems and dense control cases.

- **Online / fully-stochastic protocol:** a single pass over the data, one example per
  round; report online cumulative loss (mistakes) and the test-set performance of the
  predictor output after the pass. Identical initialization across algorithms; the step
  size `η` (and regularization multiplier `λ`) chosen on a small held-out prefix of the
  stream.
- **Text classification — Reuters RCV1 (Lewis et al. 2004):** ~800,000 articles, multiple
  labels; binary classifiers for the four top-level categories. Features are 0/1 bigram
  indicators (post stemming), ~2 million dimensions, extremely sparse (most examples have
  fewer than a few thousand non-zeros). Metric: test error rate and predictor sparsity
  (proportion of non-zero weights). The canonical sparse-feature benchmark.
- **Image ranking — ImageNet (Deng et al. 2009):** rank images per WordNet noun;
  ~10,000-dimensional sparse visterm features; ranking hinge loss
  `[1 - ⟨x, z_1 - z_2⟩]_+`. Metric: average precision and precision-at-`k`.
- **Multiclass OCR — MNIST (28×28 digit images):** kernelized features (Gaussian kernel
  on ~3000 support images, ~30,000-dimensional, dense), multiclass hinge loss; metric:
  cumulative mistakes over the pass and test error, with and without group-sparse
  `ℓ1/ℓ2`, `ℓ1/ℓ∞` regularization.
- **Income prediction — KDD census (UCI):** ~40 demographic variables quantized and
  crossed into ~4000 binary features; ~200k train / ~100k test; metric: test error and
  sparsity as a function of the fraction of the training pass seen.
- **Loss / regularizers in play:** hinge and logistic losses; composite penalties
  `φ ∈ { 0, λ||x||_1, λ||x||_2, λ||x||_∞, mixed ℓ1/ℓp }`; domains `X = R^d`, the
  `ℓ∞`-box, the `ℓ1`-ball, or `ℓ2`-ball.

## Code framework

The new rule has to plug into the same online/stochastic subgradient harness already in
use for the baselines. The substrate is only the generic machinery that exists
beforehand: a learner object that may keep state shaped like the parameter tensors and
exposes a `step()` that consumes the freshly computed subgradient, plus an outer loop
that reveals `f_t`, backpropagates to fill in `g_t`, and calls `step()`. The single empty
slot is the update rule.

```python
import torch


class OnlineSubgradientLearner:
    """Generic first-order online/stochastic learner. Owns optional state shaped
    like the parameter tensors and applies an update from the current subgradient.
    Time and memory must stay ~linear in the dimension d (no d x d matrix to store
    or invert)."""

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        # lazily-initialized state, if the rule keeps any
        self.state = {id(p): {} for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad                       # subgradient g_t for this parameter block
            state = self.state[id(p)]
            # TODO: the update rule we will design.
            #       Given the current subgradient g (and any state we choose to
            #       keep), compute the update and apply it: p += <update>(g, state)
            pass


# existing online/stochastic training loop the learner plugs into
def run_online(model, loss_fn, data_stream, learner):
    for inputs, targets in data_stream:        # one round / one fresh example
        model.zero_grad()
        outputs = model(inputs)                # forward through the existing model
        loss = loss_fn(outputs, targets)       # existing (possibly noisy) loss f_t
        loss.backward()                        # fills p.grad = g_t for every block
        learner.step()                         # apply the update rule


```

The outer loop supplies one subgradient per parameter block per round; `step()` is where
the new rule will live.
