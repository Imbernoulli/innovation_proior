## Research question

Consider a smooth saddle-point (minimax) problem

```
minimize_x  maximize_y  L(x, y),
```

where `L: R^n × R^m → R` is convex in `x` for every fixed `y` and concave in `y` for every
fixed `x`. Writing `z = (x, y)`, define the *saddle operator*

```
G(z) = [ ∇_x L(x, y) ; -∇_y L(x, y) ].
```

`G` is monotone — `⟨G(z1) - G(z2), z1 - z2⟩ ≥ 0` for all `z1, z2` — and we call `L` `R`-smooth
when `G` is `R`-Lipschitz. A point `z*` is a saddle point exactly when `G(z*) = 0`, so the
squared gradient magnitude `‖∇L(z)‖² = ‖G(z)‖²` (the sign flip on the `y`-block does not change
the norm) is a direct, always-defined measure of how close `z` is to a solution. The goal is an
algorithm whose iterate `z^k` drives `‖G(z^k)‖²` to zero as fast as possible, in the
**last iterate** (not an average and not the best-so-far), in the **unconstrained** setting
`R^n × R^m`.

Why this measure. Classical minimax theory measures suboptimality by the *duality gap*
`sup_{ỹ∈Y} L(x, ỹ) - inf_{x̃∈X} L(x̃, y)`. The gap is the natural analogue of minimization
error, and on it the optimal `O(R/k)` rate was long known (mirror-prox, dual extrapolation). The
gap needs *bounded* domains `X, Y` to be finite and does not generalize past convex-concave. The
gradient norm stays meaningful for differentiable non-convex-concave games (adversarial training,
GANs), which is exactly the regime people care about.

## Background

The field state, and the diagnostic facts that frame the problem:

- **Monotone-operator view of saddle problems (Rockafellar 1970).** A convex-concave `L` has a
  monotone saddle operator `G`; solving the minimax problem is the same as finding a zero of `G`,
  i.e. a monotone inclusion / a fixed point of a related nonexpansive map. This lets the whole
  problem be analyzed operator-side, with only monotonicity and `R`-Lipschitzness of `G` — never
  any special structure of `L` beyond that.

- **The bilinear cycling phenomenon.** The canonical hard instance is `L(x, y) = x y` on
  `R × R`, where `G(z) = [[0, 1], [-1, 0]] z` is a pure rotation. The naive simultaneous
  gradient method (descend in `x`, ascend in `y`) spirals *outward* and diverges; in continuous
  time the flow `ż = -G(z)` conserves `‖z - z*‖²` and circles the solution forever, and the
  explicit-Euler discretization grows it (Ryu, Yuan & Yin 2019, who call this the "Dirac-GAN").
  The bilinear instance is the standard stress test for minimax methods.

- **Worst-case `(δ, ν)` / smoothed-absolute-value instances.** Beyond the bilinear toy, a
  standard hard family interpolates a tiny bilinear coupling with smoothed-absolute-value terms,
  e.g. `L_{δ,ε}(x, y) = (1-δ) f_ε(x) + δ x y - (1-δ) f_ε(y)` with `f_ε` a `1`-smooth convex
  Huber-type function (Drori & Teboulle 2014 used `f_ε` as the worst case for plain gradient
  descent). These constructions are `R`-smooth with `R` essentially tight and are the canonical
  yardsticks for first-order minimax lower bounds.

- **Implicit regularization toward an anchor (Halpern 1967; Lieder 2020; Diakonikolas 2020).**
  For a *nonexpansive* map `T` (`‖T(u) - T(v)‖ ≤ ‖u - v‖`), the Halpern iteration
  `u_{k+1} = λ_{k+1} u_0 + (1 - λ_{k+1}) T(u_k)` mixes a step of `T` with a pull back toward the
  fixed starting point `u_0` (the "anchor"). With `λ_k → 0`, `Σ λ_k = ∞`, `Σ|λ_{k+1} - λ_k| < ∞`
  it converges to the fixed point of `T` nearest `u_0` in `ℓ2` — an *implicitly regularized*
  selection. With the schedule `λ_k = 1/(k+1)`, Lieder showed `‖T(u_k) - u_k‖ = 2‖u_0 - u*‖/(k+1)`,
  an `O(1/k)` rate on the fixed-point residual. Since a monotone `R`-Lipschitz `G` can be turned
  into a nonexpansive map (a resolvent / averaged operator) whose residual controls `‖G‖`, this
  residual rate transfers to the gradient norm.

- **Continuous-time / ODE reading of these dynamics.** Extragradient-type and optimistic methods
  discretize a Moreau–Yosida-regularized flow, whose `xy`-instance solution decays only like
  `exp(-λt/(1+λ²))` — slow, set by the regularization. Anchoring instead corresponds to the
  *anchored flow* `ż(t) = -G(z(t)) + (1/t)(z^0 - z(t))`, an explicit time-decaying pull toward
  the start. The two competing effects in the anchor coefficient are a *contracting* speed (the
  pull stabilizes and kills cycling) and a *vanishing* speed (the pull must die so the flow lands
  on a zero of `G`, not on `z^0`); their balance is what sets the achievable rate.

State of the art at the time: on the duality gap, `O(R/k)` was known and *optimal*. On the
squared gradient norm, the best was `O(R²/k)` and only **best-iterate** for extragradient and
optimistic methods; anchoring gave a **last-iterate** rate but only `O(R²/k^{2-2p})` and at the
price of a *diminishing* step size.

## Baselines

The prior methods a new algorithm would be measured against and would react to.

**Simultaneous gradient descent / gradient descent-ascent (simGD/GDA).**

```
z^{k+1} = z^k - α G(z^k).
```

The direct transcription of "descend in `x`, ascend in `y`." Core idea: follow the saddle
operator downhill.

**Extragradient (EG), Korpelevich 1977.**

```
z^{k+1/2} = z^k - α G(z^k),          (look-ahead / predictor)
z^{k+1}   = z^k - α G(z^{k+1/2}).    (corrector: step from z^k with the look-ahead gradient)
```

Core idea: do not commit to the gradient at the current point; take a tentative step to a
predicted point `z^{k+1/2}`, evaluate `G` *there*, and apply that corrected direction back at
`z^k`. The extra evaluation lets EG converge on monotone `R`-Lipschitz problems for `α < 1/R`,
including the bilinear instance. The load-bearing estimate is a one-step descent of
distance-to-solution: with `w = z - α G(z)` and `z^+ = z - α G(w)`,

```
‖z - z*‖² - ‖z^+ - z*‖² ≥ (1 - α²R²)‖z - w‖² = (1 - α²R²) α² ‖G(z)‖²,
```

so summing over `i = 0..k` telescopes the left side to at most `‖z^0 - z*‖²` and gives a
**best-iterate** bound `min_{i≤k} ‖G(z^i)‖² ≤ ‖z^0 - z*‖² / (α²(1-α²R²)(k+1)) = O(R²/k)`.

**Popov's algorithm / optimistic gradient descent (Popov 1980).**

```
z^{k+1} = z^k - 2α G(z^k) + α G(z^{k-1}).
```

Core idea: keep EG's anti-cycling benefit but use the *previous* gradient as the prediction, so
each step needs only one fresh gradient evaluation instead of two.

**Simultaneous gradient descent with anchoring (SimGD-A), Ryu, Yuan & Yin 2019.**

```
z^{k+1} = z^k - ((1-p)/(k+1)^p) G(z^k) + ((1-p)γ/(k+1)) (z^0 - z^k),   p ∈ (1/2, 1), γ > 0.
```

Core idea: transplant the Halpern anchor onto gradient steps — at every step add a pull back
toward the start `z^0` with a `1/(k+1)` coefficient. This is the first method to get a
**last-iterate** rate on the squared gradient norm, `O(1/k^{2-2p})`, and it tames the bilinear
cycling because the anchor damps the rotation. The gradient step size diminishes like
`(1-p)/(k+1)^p`, with `p ∈ (1/2, 1)`.

## Evaluation settings

Natural yardsticks for gradient-norm behavior:

- **Bilinear instance** `L(x, y) = x y` on `R × R`. The canonical hardest-easy convex-concave
  problem (pure rotation); the standard probe for whether a method defeats cycling. Metric:
  `‖G(z^k)‖² = ‖z^k‖²` versus iteration count, plotted on a log scale; the iterate is started
  away from the origin.
- **Worst-case `(δ, ν)` / smoothed-absolute-value instance**, e.g.
  `L_{δ,ε}(x, y) = (1-δ) f_ε(x) + δ x y - (1-δ) f_ε(y)` with `f_ε` a `1`-smooth Huber function,
  `0 < ε ≪ δ ≪ 1`, and its higher-dimensional analogue with a coordinatewise sub-gradient
  coupling. `R`-smooth with `R` near-tight; the standard instance for first-order lower bounds.
  Metric: squared gradient norm versus iterations.
- **Lagrangian of a linearly constrained quadratic program**
  `L(x, y) = ½ xᵀHx - hᵀx - ⟨Ax - b, y⟩` with `H ⪰ 0`, adopted from Ouyang & Xu (2019)'s
  duality-gap lower-bound construction. Metric: squared gradient norm versus iterations.
- Protocol: a fixed starting point `z^0` of controlled norm; a fixed step size chosen comfortably
  inside each method's theoretically convergent range; the same squared-gradient-norm quantity plotted
  for every method on a log-log axis against iteration count; in a stochastic run, each operator
  call or update line is perturbed by independent additive Gaussian noise with fixed scale `σ`,
  and the same squared-gradient-norm curve is recorded.

## Code framework

A generic single-instance saddle-point solver owns the iteration loop, the fixed starting point, and
the squared-gradient-norm logging. The problem object exposes the saddle operator `G` and, for the
stochastic setting, an additive Gaussian perturbation. The substrate is the generic machinery that
already exists: the operator, the fixed start, the iteration loop, and the per-step transition.

```python
import numpy as np


class Problem:
    """A single smooth convex-concave saddle instance min_x max_y L(x, y).
    Exposes the saddle operator G(z) = [∇_x L; -∇_y L] and (for the stochastic
    setting) an additive Gaussian perturbation. The solver only ever touches G."""

    def __init__(self, name, dim, R, sigma=0.0, seed=0):
        self.name = name          # e.g. "bilinear", "delta_nu"
        self.dim = dim            # n: z = (x, y) has length 2*dim
        self.R = R                # Lipschitz constant of G
        self.sigma = sigma        # perturbation scale (0 = deterministic run)
        self.rng = np.random.default_rng(seed)

    def grad(self, z):
        """Deterministic saddle operator G(z). (Defined per instance.)"""
        raise NotImplementedError

    def noise(self):
        """One additive Gaussian update perturbation, shape (2*dim,)."""
        return self.rng.normal(0.0, self.sigma, size=2 * self.dim)


def init_state(problem, initial_z, hyperparameters):
    """Initialize the method state from the fixed starting point."""
    raise NotImplementedError


def step(state, problem, hyperparameters):
    """Advance one iteration of the method, returning the new state and the
    iterate whose squared gradient norm is measured."""
    raise NotImplementedError


def get_hyperparameters(problem_name):
    """Return the per-instance constants the method uses."""
    raise NotImplementedError


# generic single-instance driver
def run(problem, z0, num_iters, hyperparameters):
    state = init_state(problem, z0, hyperparameters)
    for k in range(num_iters):
        state, z_measured = step(state, problem, hyperparameters)
        g_measured = problem.grad(z_measured)
        log_gradient_norm_sq(float(np.dot(g_measured, g_measured)))   # ‖G(z)‖²
    return state
```

The driver supplies the saddle operator and the fixed start, and records the squared saddle-operator
norm at the returned last iterate.
