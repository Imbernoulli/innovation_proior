# REINFORCE: score-function gradient following for stochastic networks

## Problem

A network of stochastic units receives only a scalar reinforcement `r` per trial, broadcast to
all units. There is no target, no per-unit error, and the map from (input, output) to `r` is an
unknown, possibly non-differentiable black box. Goal: a **local, model-free** weight-update rule
that provably climbs the expected reinforcement `E{r|W}` — without backpropagating through the
environment or storing an estimate of its gradient.

## Key idea (the score-function / likelihood-ratio trick)

The gradient of an expectation taken over a parameterized distribution can itself be written as an
expectation over that distribution, by differentiating the *log-probability of the sampled output*
instead of the (unavailable) reward:

  ∂E{r|W,x^i}/∂w_ij = Σ_ξ E{r | y_i=ξ} ∂g_i/∂w_ij
                    = Σ_ξ E{r | y_i=ξ} g_i · ∂ln g_i/∂w_ij
                    = E{ r · ∂ln g_i/∂w_ij }.

So a single sampled `(output, reward)` pair, with the reward used as a bare scalar multiplier on
the gradient of the log-likelihood of the output, is an **unbiased estimate of the gradient of
expected reward**. The factor `e_ij = ∂ln g_i/∂w_ij` — the **characteristic eligibility** — is the
only place the unit's own structure enters, and it is computable locally in closed form.

## The REINFORCE rule

For every weight, at the end of each trial:

  **Δw_ij = α_ij (r − b_ij) e_ij**,   `e_ij = ∂ln g_i/∂w_ij`,

with `α_ij ≥ 0` a rate factor, and `b_ij` a **reinforcement baseline** that is conditionally
independent of the current output `y_i` given `(W, x^i)`. ("REward Increment = Nonnegative Factor
× Offset Reinforcement × Characteristic Eligibility.")

## Unbiasedness theorem (proved)

**Baseline invariance (the `∇1 = 0` fact).** Since `g_i` is a probability distribution,
`Σ_ξ ∂g_i/∂w_ij = ∂/∂w_ij (Σ_ξ g_i) = ∂/∂w_ij (1) = 0`. Hence the eligibility has mean zero,
`E{e_ij | W, x^i} = Σ_ξ g_i (1/g_i) ∂g_i/∂w_ij = 0`, and for any output-independent `b`,
`E{b·e_ij | W,x^i} = E{b|W,x^i} Σ_ξ ∂g_i/∂w_ij = 0`. **Any admissible baseline leaves the gradient estimate unbiased**;
it only changes the variance. (`b ≈ E{r}` — reinforcement comparison — centers the multiplier on
the reward's surprise and reduces variance.)

**Per-weight unbiasedness.** Fixing `y_i = ξ` screens off `w_ij`'s influence on `r`, so
`E{r | y_i=ξ}` is free of `w_ij`; differentiating `E{r|W,x^i} = Σ_ξ E{r|y_i=ξ} g_i` then gives, with
the baseline term killed by the zero-mean eligibility,

  E{Δw_ij | W, x^i} = α_ij ∂E{r|W,x^i}/∂w_ij.

**Averaging over inputs.** `x^i` is upstream of `w_ij`, so `Pr{x^i=x|W}` is free of `w_ij`; hence
`∂E{r|W}/∂w_ij = Σ_x ∂E{r|W,x^i=x}/∂w_ij Pr{x^i=x|W}`, and averaging the update the same way gives

  E{Δw_ij | W} = α_ij ∂E{r|W}/∂w_ij.

**Theorem 1.** For any REINFORCE algorithm,

  E{ΔW|W}ᵀ ∇_W E{r|W} = Σ_{ij} α_ij ( ∂E{r|W}/∂w_ij )² ≥ 0,

with equality iff `∇_W E{r|W} = 0` (when all `α_ij > 0`); and if `α_ij = α` (constant) then
**E{ΔW|W} = α ∇_W E{r|W}** — exact gradient ascent on expected reward, in expectation. Equivalently,
`(r − b_ij) ∂ln g_i/∂w_ij` is an unbiased estimate of `∂E{r|W}/∂w_ij`.

## Eligibilities for standard units

- **Bernoulli** (`p = Pr{y=1}`): `∂ln g/∂p = +1/p` when `y=1`, and
  `∂ln g/∂p = −1/(1−p)` when `y=0`; both branches equal `(y − p)/(p(1−p))`. With `b=0, α=ρ p(1−p)`:
  `Δp = ρ r (y−p)`; for `r ∈ {0,1}` this is the two-action linear reward-inaction `L_{R-I}`.
- **Bernoulli semilinear** (`p = f(s)`, `s = Σ_j w_ij x_j`): `∂ln g/∂w_ij = [(y−p)/(p(1−p))] f'(s) x_j`;
  for the logistic `f'=p(1−p)`, this collapses to `(y−p)x_j`, giving `Δw_ij = α r (y_i−p_i) x_j` — the
  associative reward-inaction `A_{R-I}` rule (the `λ=0` case of `A_{R-P}`), now exhibited as a
  gradient-follower.
- **Gaussian** (`y ~ N(μ,σ²)`): from
  `ln g = −½ln(2π) − lnσ − (y−μ)²/(2σ²)`,
  `∂ln g/∂μ = (y−μ)/σ²`, and
  `∂ln g/∂σ = −1/σ + (y−μ)²/σ³ = ((y−μ)²−σ²)/σ³`, giving
  `Δμ = α_μ(r−b)(y−μ)/σ²`, `Δσ = α_σ(r−b)((y−μ)²−σ²)/σ³`. `σ` is a controllable exploration width:
  rewarded near-the-mean samples shrink it, rewarded far samples grow it.
- **Exponential family** (`g = exp[Q(μ,…)y + D(μ,…) + S(y)]`, `μ` the mean): `∂ln g/∂μ = (y−μ)/σ²`
  for the whole family. Write `∂ln g/∂μ = a y + c`; the zero-mean score gives `aμ+c=0`, while
  `1 = ∂μ/∂μ = Σ_y (y−μ)∂g/∂μ = E[(y−μ)(ay+c)] = aσ²`, so `a=1/σ²` and `c=−μ/σ²`.
  Thus Bernoulli, Gaussian, Poisson, exponential, and the rest share the
  `(output − mean)/variance` eligibility form.

## Episodic REINFORCE (delayed reward)

Unfold the net over `k` steps into an acyclic net with tied weights `w_ij^t = w_ij`. By the chain
rule `∂E{r|W}/∂w_ij = Σ_t ∂E{r|W*}/∂w_ij^t`, so

  **Δw_ij = α_ij (r − b_ij) Σ_{t=1}^k e_ij(t)**,

and the same inner-product argument gives Theorem 2 (identical conclusion). The eligibility sum is
accumulated online from the unit's own operation before `r` arrives — one accumulator per weight —
so credit is spread (uniformly) over the episode.

## Minimal local implementation

```python
import numpy as np

def logistic(s):
    return 1.0 / (1.0 + np.exp(-s))

class BernoulliLogisticUnit:
    """p = logistic(w·x), y ~ Bernoulli(p); trained by REINFORCE with a comparison baseline."""
    def __init__(self, n_in, alpha=0.1, gamma=0.1):
        self.w = np.zeros(n_in)
        self.alpha, self.gamma, self.rbar = alpha, gamma, 0.0

    def forward(self, x):
        self.x = x
        self.p = logistic(self.w @ x)
        self.y = float(np.random.rand() < self.p)     # sample stochastic output
        return self.y

    def eligibility(self):
        return (self.y - self.p) * self.x              # (y-p)x  for Bernoulli-logistic

    def learn(self, r):
        dw = self.alpha * (r - self.rbar) * self.eligibility()   # Δw = α(r-b)e, unbiased
        self.w += dw
        self.rbar = self.gamma * r + (1 - self.gamma) * self.rbar  # update baseline after use
        return dw


class GaussianUnit:
    """y ~ N(mu, sigma^2); mu, sigma adapted by REINFORCE."""
    def __init__(self, alpha=0.01, gamma=0.1):
        self.mu, self.sigma = 0.0, 1.0
        self.alpha, self.gamma, self.rbar = alpha, gamma, 0.0

    def forward(self):
        self.y = self.mu + self.sigma * np.random.randn()
        return self.y

    def learn(self, r):
        b, mu, s = self.rbar, self.mu, self.sigma
        self.mu    += self.alpha * s**2 * (r - b) * (self.y - mu) / s**2
        self.sigma += self.alpha * s**2 * (r - b) * ((self.y - mu)**2 - s**2) / s**3
        self.sigma  = max(self.sigma, 1e-3)
        self.rbar   = self.gamma * r + (1 - self.gamma) * self.rbar


def episodic_update(eligibility_steps, r, alpha=0.1, baseline=0.0):
    # eligibility_steps: per weight, list of per-step e_ij(t) accumulated online over the episode.
    return alpha * (r - baseline) * np.sum(eligibility_steps, axis=0)   # Δw = α(r-b)Σ_t e(t)
```
