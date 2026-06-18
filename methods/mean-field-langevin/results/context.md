# Context: feature learning for low-dimensional targets with two-layer networks

## Research question

A two-layer network is asked to fit a target that depends on its high-dimensional input only
through a few directions:

```
y = g(U x) + xi,   x in R^d,   U in R^{k x d} with orthonormal rows,   k << d,
```

where both the projection `U` (the `k`-dimensional relevant subspace) and the scalar link
`g : R^k -> R` are unknown, and `xi` is mean-zero subGaussian noise. The input has zero mean
and covariance `Sigma`; the isotropic case is `Sigma = I_d`. Recovering the model means
recovering *both* `U` and `g` from `n` i.i.d. samples.

Counting parameters, `g` lives on `k = O(1)` directions, so there are about `k d = O(d)`
numbers to estimate, and on isotropic data the information-theoretically optimal sample size
is `n ~ d`. The pressing question is whether a *standard first-order training procedure* —
gradient descent on a two-layer net with the usual regularization, not a hand-built
spectral estimator — can reach that `n ~ d` floor for an *arbitrary* link `g`, or whether
first-order training is structurally stuck far above it. A solution has to (i) actually find
the subspace `U` (escape the flat region around a random initialization where the gradient
signal is weak), and (ii) do so with sample complexity that does not blow up as the link `g`
becomes more "high-order".

## Background

**Single-index hardness is graded by the information exponent.** For a single relevant
direction (`k = 1`), Ben Arous, Gheissari & Jagannath (2021) showed that the difficulty of
online SGD on the correlation/squared loss is set by the *information exponent* `s` of the
link: writing the population correlation as a function of the overlap between a weight and the
true direction and expanding it about the uninformative equator, `s` is the degree of the
first nonvanishing term in that expansion (equivalently, the index of the first nonzero
Hermite coefficient of `g`). Their thresholds for recovering the direction are `n ~ d` when
`s = 1`, `n ~ d log d` when `s = 2`, and `n ~ d^{s-1}` when `s >= 3`. The mechanism is a
search phase: at a random start the overlap is `O(1/sqrt(d))`, and when `s >= 3` the drift
pulling the weight toward the true direction scales like `overlap^{s-1}`, which is minuscule,
so the iterate diffuses for a very long time before the signal takes over. High information
exponent means a vanishingly weak gradient at initialization.

**Multi-index hardness is graded by the leap.** Abbe, Boix-Adsera & Misiakiewicz (2023)
extended this to several relevant directions with isotropic data, defining the *leap
complexity* `Leap(f)` — how large a jump in the "support" of the Hermite expansion the link
forces, capturing hierarchical/staircase structure where some directions can only be found
after others. They proved SGD on a two-layer net learns such targets in about
`d^{max(Leap(f), 2)}` steps, through a *saddle-to-saddle* dynamics: training sits near a
saddle, slowly aligns one (group of) direction(s), drops to a lower saddle, and repeats. For
`Leap >= 3` this is again strongly super-linear in `d`. Statistical-query lower bounds predict
the same `d^{Theta(s)}` wall for this whole family of gradient-based learners.

So the prevailing picture is a gap: the information-theoretic floor is `n ~ d`, but the sample
complexity of gradient-trained networks is governed by an *exponent of the link* and can sit
arbitrarily far above the floor. There exist sample-optimal estimators that ignore the
exponent, but they are not standard first-order training — they are special-purpose
(spectral/method-of-moments) or require exponential compute.

**Lifting the network to a distribution over neurons.** A separate line (Mei, Montanari &
Nguyen 2018; Chizat & Bach 2018; Rotskoff & Vanden-Eijnden) studies wide two-layer networks
in the *mean-field regime*. A width-`m` network `yhat_m(x;W) = (1/m) sum_j Psi(x;w_j)` and the
norm regularizer `(1/m) sum_j ||w_j||^2` are both invariant to permuting the neurons, so they
depend on `W` only through the empirical measure `mu = (1/m) sum_j delta_{w_j}` over neuron
weights:

```
yhat(x; mu) = integral Psi(x; w) dmu(w),    R(mu) = integral ||w||^2 dmu(w).
```

In this lifted view the data-fit functional is *convex* in `mu` (the prediction is linear in
`mu` and the per-example loss `rho` is convex), even though the finite-width loss in `W` is
non-convex. As `m -> infinity` the empirical measure of weights under gradient training
converges (propagation of chaos) to a deterministic measure `mu_t` evolving by a
Wasserstein gradient flow / nonlinear Fokker–Planck equation. The appeal is global: convexity
in measure space means there are no spurious local minima to get stuck in *as a measure*. The
limitation of this line as it stood is that its convergence statements were largely
*qualitative* — existence of a limit, no rate, and no learnability/generalization guarantee
tying the measure-space optimum back to recovering `U` and `g` from finitely many samples.

**Convex optimization in measure space needs a controlling constant.** To get *rates*, one
adds entropy. Define the relative entropy and relative Fisher information of `mu` against a
reference `nu`,

```
H(mu | nu) = integral ln(dmu/dnu) dmu,   I(mu | nu) = integral || grad ln(dmu/dnu) ||^2 dmu.
```

Nitanda, Wu & Suzuki (2022) and Chizat (2022) analyzed the entropy-regularized free energy
`F_beta(mu) = J(mu) + (1/beta) H(mu | tau)` (with `tau` a base measure and `beta` an inverse
temperature) and identified the object that controls convergence: the *proximal Gibbs
distribution* of the current measure,

```
p_mu  proportional to  exp(-beta * J'[mu]),
```

where `J'[mu]` is the first variation of `J` at `mu` (the function whose integral against
`nu - mu` gives the directional derivative of `J`). The role of `p_mu` is that it "fills the
duality gap": the gap between `F_beta(mu)` and its minimum is controlled by how far `mu` is
from `p_mu`. If `p_mu` obeys a *log-Sobolev inequality* (LSI) — `H(mu | p_mu) <= (C_LSI/2)
I(mu | p_mu)` for all `mu`, the measure-space analogue of strong convexity — uniformly along
the trajectory, then the free energy decays exponentially,

```
F_beta(mu_t) - F_beta(mu*_beta) <= exp(-2t / (beta C_LSI)) * (F_beta(mu_0) - F_beta(mu*_beta)).
```

`C_LSI` is the controlling constant: convergence speed is governed entirely by it. A standard
way to certify an LSI is the Holley–Stroock perturbation argument (a bounded perturbation of a
strongly log-concave potential keeps an LSI, with the constant inflated by `exp` of the
perturbation range), and on a compact manifold the Bakry–Émery criterion turns a lower bound
on Ricci curvature directly into an LSI constant. A known hazard is that in Euclidean space
`C_LSI` can be *exponential* in `beta` in the worst case (Menz–Schlichting 2014), so the
regime where `beta` must be large is exactly where the controlling constant is dangerous.

**A worked discrete-time guarantee for the lifted dynamics.** Suzuki, Wu & Nitanda (2023)
made the analysis finite by handling time, particle, and stochastic-gradient discretization
together, proving a one-step decay of the finite-width free energy of the form
`F^m(mu_{l+1}) - F(mu*) <= exp(-eta / (2 beta C_LSI)) (F^m(mu_l) - F(mu*)) + eta * A`, with the
additive `A` collecting `O(eta + 1/m)` discretization error. This converts the continuous LSI
contraction into a guarantee for an actually-runnable iteration.

**Where the lifted approach had only been pushed.** Prior learnability results in the lifted
regime were confined to special targets and special data — `k`-sparse parity on the hypercube,
or single-index models with specific links on isotropic Gaussian inputs — and to reach
sample-optimality they spent compute exponential in the ambient dimension. Learning *general*
`g` of low-dimensional projections, under broad subGaussian data with general covariance,
through a standard training procedure, was open.

## Baselines

**Online SGD on a two-layer net (correlation / squared loss).** Initialize weights at random,
descend the empirical loss by minibatch gradient steps, possibly with one pass over the data.
For single-index targets the recovery time and sample complexity track the information
exponent `s` (`n ~ d^{s-1}` for `s >= 3`); for multi-index targets they track the leap
(`d^{max(Leap, 2)}` steps), proceeding saddle-to-saddle. **Gap:** the cost is dictated by an
exponent of the link rather than by the dimension count `kd ~ d`; for high-order links it sits
far above the `n ~ d` floor, because at a random start the gradient signal toward the relevant
subspace is `overlap^{s-1}` with `overlap = O(1/sqrt(d))`, so the search phase is long. This
is the vanilla `sgd_mlp` reference.

**Two-stage / layer-wise fitting.** First fit the first layer by spherical SGD on a
correlation loss (so each neuron's direction aligns with a relevant direction), holding the
output layer fixed; then solve the output layer in closed form by ridge regression on the
post-activation features (Bietti, Bruna & Sanford 2022; Dandi et al. 2024). Decoupling the two
layers and giving the output a closed-form solve sharpens the analysis of the first "giant
step". **Gap:** it still relies on the first-layer correlation step to surface the directions,
so its hard cases inherit the same exponent-governed search-phase difficulty, and it is a
staged, hand-structured procedure rather than one homogeneous training rule.

**Fixed-grid / random-features regression.** Freeze the first layer at its random
initialization and train only the (convex) output layer — random features. Sidesteps
non-convex first-layer dynamics entirely. **Gap:** with the first layer frozen, the network
performs *no feature learning*; to approximate a function of unknown low-dimensional
projections from a fixed random representation, the number of features (and hence the compute)
scales exponentially with the ambient dimension `d`, with no adaptivity to any low-dimensional
structure in the covariance.

**Lifted-measure training on special targets.** Mean-field training of a wide two-layer net,
analyzed for `k`-sparse parity on the hypercube or single-index models on isotropic data.
Establishes that the convex measure-space view can reach `n ~ d` despite a large leap, on
those targets. **Gap:** restricted to those data/target classes, and the compute to attain
sample-optimality is exponential in `d`.

## Evaluation settings

- **Targets.** Multi-index regression `y = g(Ux) + xi` with `U in R^{k x d}` orthonormal,
  `k = O(1)`, and a chosen link `g`. A canonical hard family takes `g` built from degree-`>=3`
  Hermite polynomials (information/leap exponent `3`), e.g. a sum of third Hermite polynomials
  `He_3(z) = z^3 - 3z` over the relevant coordinates. A simple single-index instance is the
  second Hermite link `g(z) = (z^2 - 1)/sqrt(2)`.
- **Inputs.** `x ~ N(0, Sigma)`. Isotropic `Sigma = I_d`, and structured (spiked or
  power-law-spectrum) covariances where the relevant directions align with high-variance
  eigendirections. Ambient `d` in the tens to low hundreds.
- **Network.** Two-layer `Linear(d, m) -> activation -> Linear(m, 1)`, small width `m`
  (tens). A standard choice freezes the second layer at fixed `+-1` values (half positive,
  half negative) so only the first-layer directions are learned.
- **Protocol.** Sweep the number of samples `n` and a measure of the effective
  low-dimensionality of the covariance; train for a fixed iteration budget; average over a few
  seeds (data realization + initialization).
- **Metrics.** Test mean-squared error on a large fresh test set; the train/test
  generalization gap; and the quality of subspace recovery — how close the leading subspace of
  the trained first-layer weights is to the true `U`.

## Code framework

The training plugs into a generic two-layer regression harness. The data pipeline, the fixed
second layer, the predictor, the squared loss, and the outer loop already exist; what is *not*
fixed is the per-step update of the first-layer weights — that rule is the open slot.

```python
import torch


def predict(X, W, a, phi):
    """Two-layer net: hidden = phi(X W^T), output = hidden @ a (a is the fixed second layer)."""
    return phi(X @ W.T) @ a


def data_gradient(X, y, W, a, phi, phiprime):
    """Gradient of the per-batch squared loss w.r.t. the first-layer weights W."""
    residual = predict(X, W, a, phi) - y                    # [n]
    backprop = phiprime(W @ X.T) * a.reshape(-1, 1)         # [m, n]
    return (backprop * residual.reshape(1, -1)) @ X / X.shape[0]   # [m, d]


def init_first_layer(m, d):
    """Initialize the m hidden-neuron weight rows."""
    W = torch.randn(m, d)
    # TODO: any structure we choose to impose on the initial neuron weights.
    return W


def update_step(W, X, y, a, phi, phiprime, lr, hparams):
    """One update of the first-layer weights from one batch.

    The data gradient is available via data_gradient(...). What else the step does to W
    -- any regularization drift, any stochasticity, any constraint on the neurons --
    is exactly what we are here to design.
    """
    # TODO: the first-layer update rule we will design.
    raise NotImplementedError


def train(X, y, W, a, phi, phiprime, n_iters, lr, hparams):
    for _ in range(n_iters):
        W = update_step(W, X, y, a, phi, phiprime, lr, hparams)
    return W
```

The second layer `a` is fixed; the predictor, loss, and loop are settled. The single empty
slot is `update_step` (and whatever it needs from `init_first_layer`).
