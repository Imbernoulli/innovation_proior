# The Bootstrap Particle Filter (Sequential Monte Carlo)

## Problem

For a discrete-time state-space model
$$x_k = f_{k-1}(x_{k-1}, w_{k-1}), \qquad y_k = h_k(x_k, v_k),$$
with $f, h$ possibly nonlinear and $w, v$ noises of known (not necessarily Gaussian) density,
compute the filtering posterior $p(x_k \mid y_{1:k})$ online, one bounded-cost step per measurement.
Off the linear-Gaussian case (where the Kalman filter is exact because a Gaussian stays Gaussian),
this posterior has no closed form, and the extended Kalman filter — which forces a single Gaussian
— distorts multimodal/curved posteriors and can diverge (e.g. bearings-only tracking).

## Key idea

Represent the posterior not as a function or a grid but as a cloud of $N$ weighted random samples
(**particles**); its empirical measure approximates the density and converges to it as $N\to\infty$,
with Monte Carlo error $O(1/\sqrt N)$ independent of dimension. Propagate the cloud through the exact
Bayesian recursion:

- **Predict (Chapman–Kolmogorov), realised by simulation.** Because $x_k=f_{k-1}(x_{k-1},w_{k-1})$ is
  a deterministic map given the noise, pushing each particle through $f$ with a fresh draw
  $w_{k-1}(i)\sim p(w_{k-1})$ produces $x_k^*(i)=f_{k-1}(x_{k-1}(i),w_{k-1}(i))$, a sample from the
  predictive $p(x_k\mid y_{1:k-1})$. No density evaluation needed.
- **Update (Bayes) as a weighted bootstrap.** Since $p(x_k\mid y_{1:k})\propto g(y_k\mid x_k)\,
  p(x_k\mid y_{1:k-1})$, weight each predicted particle by the likelihood,
  $W(i)\propto g(y_k\mid x_k^*(i))$, then resample $N$ times with replacement using $\{W(i)\}$. By
  the Smith–Gelfand (1992) sampling–resampling result, the resampled cloud is an (unweighted) sample
  from the posterior; the intractable normaliser $p(y_k\mid y_{1:k-1})$ cancels.

## Why resampling is essential (degeneracy and the fix)

Carrying weights forward without resampling is **sequential importance sampling (SIS)**. With
factorised proposal $q_k(x_{1:k})=q_1(x_1)\prod_j q_j(x_j\mid x_{1:j-1})$ and target
$\gamma_k=p(x_{1:k},y_{1:k})$, the unnormalised weight telescopes:
$$w_k = w_{k-1}\,\alpha_k, \qquad
\alpha_k = \frac{\gamma_k}{\gamma_{k-1}\,q_k(x_k\mid x_{1:k-1})}
        = \frac{g(y_k\mid x_k)\,f(x_k\mid x_{k-1})}{q_k(x_k\mid x_{1:k-1})}.$$
The weight is a product of $k$ factors, so its variance grows — typically **exponentially** — in $k$.
Toy check (target $\prod_j N(x_j;0,1)$, proposal $\prod_j N(x_j;0,\sigma^2)$):
$$\frac{\mathbb V[\hat Z_k]}{Z_k^2}=\frac1N\!\left[\Big(\tfrac{\sigma^4}{2\sigma^2-1}\Big)^{k/2}-1\right],
\qquad \text{finite for }\sigma^2>\tfrac12,$$
and on that finite side
$$\tfrac{\sigma^4}{2\sigma^2-1}=1+\tfrac{(\sigma^2-1)^2}{2\sigma^2-1}>1 \quad \text{for }\sigma^2\neq1.$$
At $\sigma^2{=}1.2$, $k{=}1000$ this needs $N\approx2\times10^{23}$ particles. In practice a single
particle ends up holding nearly all the weight; the cloud collapses. The collapse is monitored by the
**effective sample size**
$$\text{ESS}=\Big(\textstyle\sum_i W_i^2\Big)^{-1}\in[1,N],$$
which crashes toward $1$.

**Resampling cuts the product.** Drawing offspring $N_k^{1:N}\sim\text{Multinomial}(N,W_k^{1:N})$ and
resetting weights to $1/N$ deletes low-weight particles, duplicates high-weight ones, and is unbiased
($\mathbb E[N_k^i\mid W]=N W_k^i$). It resets the system to the target at each step, replacing the
single global proposal in the variance with the per-step resampled proposal, so on the same toy
$$\frac{\mathbb V_{\text{SMC}}[\hat Z_k]}{Z_k^2}\approx
\frac kN\!\left[\Big(\tfrac{\sigma^4}{2\sigma^2-1}\Big)^{1/2}-1\right]$$
— **linear** in $k$ ($N\approx10^4$ for the same accuracy). Under the model's forgetting/mixing the
filtering-marginal variance grows at order $C k/N$. Costs: extra immediate Monte Carlo variance,
and path-space degeneracy (early states lose distinct values) — irrelevant for filtering, harmful for
smoothing. Resample adaptively, only when ESS $< N/2$.

## The bootstrap proposal

Choosing the proposal to be the **transition prior**, $q_k=f(x_k\mid x_{k-1})$, cancels $f$ and makes
$\alpha_k=g(y_k\mid x_k)$ — weight equals likelihood. This is the bootstrap filter; it needs only to
*sample* $f$ and *evaluate* $g$. The variance-minimising proposal is
$q_k^{\text{opt}}=p(x_k\mid y_k,x_{k-1})\propto g(y_k\mid x_k)f(x_k\mid x_{k-1})$ (then
$\alpha_k=p(y_k\mid x_{k-1})$, independent of $x_k$), but it is generally not samplable and its
normaliser is intractable. The prior proposal's weakness: it ignores $y_k$, so a sharp likelihood far
from the predictive prior (small overlap) yields uneven weights and faster degeneracy. Remedies:
**roughening** — after resampling add jitter $\epsilon\sim N(0,J_k)$ with per-component standard
deviation $\sigma=K\,E\,N^{-1/d}$ ($E$ = sample range, $d$ = state dimension, $K\approx0.2$) to break
duplicate ties when process noise is too small; and optional **prior editing** — reject prior samples
certain to fall outside the next likelihood.

## Algorithm

Initialise: draw $x_1(i)\sim p(x_1)$, weights $1/N$, and feed these particles directly into the
first update with $y_1$ by starting at the weighting step. For each later $k$:
1. **Predict:** $x_k^*(i)=f_{k-1}(x_{k-1}(i),w_{k-1}(i))$, $w_{k-1}(i)\sim p(w_{k-1})$.
2. **Weight:** $\tilde W(i)\propto W(i)\,g(y_k\mid x_k^*(i))$; normalise.
3. **Resample if** ESS $<N/2$: $x_k\leftarrow$ resample$(x_k^*,W)$ (systematic), weights $\to1/N$;
   roughen.
4. **Estimate:** any summary as a (weighted) average over the cloud.

Requirements: $p(x_1)$ samplable, $g(y_k\mid x_k)$ a known functional form, $p(w_k)$ samplable. No
linearity or Gaussian assumption; embarrassingly parallel across particles.

## Reference implementation

```python
import numpy as np

def systematic_resample(weights):
    """Low-variance O(N) resampling: one uniform offset, N equally spaced points up the weight CDF."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    N = weights.size
    positions = (np.random.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions, side="left")


class BootstrapParticleFilter:
    """Filtering posterior p(x_k | y_1:k) as a cloud of N particles."""

    def __init__(self, model, N, ess_threshold=0.5, roughen_K=0.2):
        self.model = model        # sample_prior, propagate (f + fresh noise), log_likelihood (g)
        self.N = N
        self.ess_threshold = ess_threshold * N
        self.K = roughen_K
        self.x = None; self.w = None

    def initialise(self):
        self.x = self.model.sample_prior(self.N)        # draws from p(x_1)
        self.w = np.full(self.N, 1.0 / self.N)

    def predict(self):
        self.x = self.model.propagate(self.x)           # Chapman-Kolmogorov, realised by simulation

    def update(self, y):
        logw = np.log(self.w + 1e-300) + self.model.log_likelihood(y, self.x)  # incremental weight = likelihood
        max_logw = np.max(logw)
        if not np.isfinite(max_logw):
            raise ValueError("all particle likelihoods are zero")
        w = np.exp(logw - max_logw)
        total = w.sum()
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("particle weights are not normalisable")
        self.w = w / total                              # Bayes denominator cancels in normalisation
        ess = 1.0 / np.sum(self.w ** 2)                 # ESS = (sum W_i^2)^{-1}
        if ess < self.ess_threshold:                    # adaptive resampling
            self.x = self.x[systematic_resample(self.w)]
            self.w = np.full(self.N, 1.0 / self.N)      # cut the weight product
            self._roughen()

    def _roughen(self):
        d = self.x.shape[1]
        rng = self.x.max(axis=0) - self.x.min(axis=0)
        sigma = self.K * rng * self.N ** (-1.0 / d)     # scales like an N-point grid spacing
        self.x = self.x + np.random.randn(*self.x.shape) * sigma

    def estimate(self):
        return np.average(self.x, axis=0, weights=self.w)


def run_bootstrap_filter(model, observations, N=1000):
    pf = BootstrapParticleFilter(model, N); pf.initialise()
    out = []
    for k, y in enumerate(observations):
        if k > 0:
            pf.predict()
        pf.update(y); out.append(pf.estimate())
    return out
```
