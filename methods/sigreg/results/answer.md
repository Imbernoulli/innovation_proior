# SIGReg, distilled

SIGReg (Sketched Isotropic Gaussian Regularization) is an anti-collapse regularizer for
joint-embedding self-supervised learning. It pushes a batch of embeddings toward an
isotropic Gaussian `N(0, I)` by *slicing*: it projects the embeddings onto many random
unit directions and, on each one-dimensional projection, runs a differentiable Gaussianity
test (the Epps-Pulley characteristic-function test), then averages the per-direction
statistics. Combined with an invariance (predictability) term, it removes the need for
stop-gradients, teacher-student / EMA networks, negatives, and whitening, at linear `O(N)`
time and memory in the batch size and with provably bounded gradients and curvature.

## Problem it solves

The predictability objective of joint-embedding SSL has trivial minima — complete collapse
(constant encoder) and dimensional collapse (low-rank embeddings). The anti-collapse term
must rule those out *and* be: identifiable (not satisfiable by a degenerate configuration),
`O(N)` and data-parallel-friendly, gradient/curvature-bounded for stable large-scale
training, and controlled by a single trade-off hyperparameter.

## Key idea

1. **Target distribution = isotropic Gaussian, derived not assumed.** Because downstream
   tasks are unknown, choose the embedding geometry that minimizes worst-case risk.
   - *Linear probe:* ridge bias `-lambda (Z^T Z + lambda I)^{-1} beta_true` is larger along
     weak eigendirections for anisotropic covariance, and the unregularized variance
     `sigma^2 sum_j 1/lambda_j` is minimized (Jensen) when all eigenvalues are equal ⇒
     **isotropy**.
   - *Nonlinear probe (radius-kNN / kernel):* leading integrated squared bias depends on the
     Fisher-information functional `J(p) = ∫ ||∇log p||^2 p dx`. At fixed covariance `Σ`, the
     Cramér-Rao bound gives `J(p) ≥ tr(Σ^{-1})`, with equality **iff `p` is Gaussian**
     (score affine ⇒ Gaussian). Minimizing `tr(Σ^{-1}) = sum_i 1/λ_i` under any scalar
     covariance constraint forces `Σ = sI` ⇒ **isotropic Gaussian**, WLOG `N(0, I)`.

2. **Distribution matching as a sliced hypothesis test.** Test `H0: P_θ = Q`. Direct
   multivariate normality tests (BHEP / Henze-Zirkler) are `O(N^2)` pairwise sums and do not
   shard across devices. **Hyperspherical Cramér-Wold:** `X =d Y` iff `<u,X> =d <u,Y>` for
   all unit `u` (proof: `φ_X(t) = φ_X(s u) = φ_Y(s u) = φ_Y(t)` ⇒ uniqueness ⇒ `X =d Y`).
   So match every unit-direction projection to `N(0,1)`. For a *loss*, **average** the
   per-direction statistic over a sampled set `A` (the `max` of the consistency theorem
   gives sparse gradients), and **resample `A` every step** for sphere coverage.

3. **The 1-D test = Epps-Pulley (characteristic function).**
   - *Moments* (Jarque-Bera / extended): a finite number of moments does not determine the
     distribution (non-identifiable shortcuts), and `K → ∞` makes gradients blow up — the
     moment-objective gradient is a degree-`(k-1)` polynomial in the sample, `~|X_i|^{k-1}`,
     unbounded for `k ≥ 2`. Rejected.
   - *CDF* (Cramér-von Mises, Anderson-Darling, Watson, KS, Shapiro-Wilk): identifiable but
     require **sorting** — non-differentiable and breaks data-parallel SGD (global sync); KS
     uses `L∞` (sparse gradients). Rejected.
   - *Characteristic function* (Epps-Pulley): the empirical CF `φ̂(t) = (1/n) Σ_j e^{i t X_j}`
     is a differentiable average that all-reduces across devices, is identifiable (full
     Fourier content), and is **provably gradient/curvature-bounded**. Chosen.

## Final objective

Per direction `a` (unit), with target CF `φ(t) = e^{-t^2/2}` (the CF of `N(0,1)`) and a
Gaussian window `w(t) = e^{-t^2/σ^2}`; the canonical implementation reuses the target CF as
the window, i.e. `w(t) = e^{-t^2/2}`:

```
EP(a) = N ∫ |φ̂_{a^T Z}(t) - e^{-t^2/2}|^2 w(t) dt
SIGReg = (1/|A|) Σ_{a ∈ A} EP(a)          # average over directions, A resampled each step
```

Total joint-embedding loss (SIGReg as the anti-collapse half):
`L = (1 - λ) · L_pred + λ · SIGReg`, with prediction loss
`L_pred = (1/V) Σ_{v'} || μ_n - z_{n,v'} ||^2`, `μ_n = (1/V_g) Σ_{v=1}^{V_g} z_{n,v}`.
Single hyperparameter `λ`.

## Stability theorem (why Epps-Pulley)

Write the weight as `w_s(t) = e^{-s^2 t^2}` (the window `e^{-t^2/σ^2}` is this with
`s^2 = 1/σ^2`) and estimator `D = ∫ w_s |φ̂_N - φ_G|^2 dt`, using only `|φ̂_N| ≤ 1`,
`|φ_G| ≤ 1`, `|e^{itX}| = 1` and `∂φ̂_N/∂X_i = (1/N) i t e^{itX_i}`:

```
|∂D/∂X_i|  ≤ (4/N) ∫ w_s(t)|t| dt   = 4/(N s^2)     = 4σ^2/N        (since 1/s^2 = σ^2)
|∂²D/∂X_i²| ≤ (C/N) ∫ w_s(t) t^2 dt = C√π/(2 N s^3) = C√π σ^3/(2N)  (since 1/s^3 = σ^3)
```

using `∫ e^{-s^2 t^2}|t| dt = 1/s^2` and `∫ e^{-s^2 t^2} t^2 dt = √π/(2 s^3)`. Bounded for
*any* input distribution; moments cannot achieve this. Chain rule:
`||∇_θ D|| ≤ (4σ^2/N) Σ_i ||a^T ∇_θ f_θ(x_i)||`; multiplying by `N` for the classical
Epps-Pulley statistic rescales this bound by the same fixed factor.

## Curse of dimensionality and minibatch bias

- **Smoothness:** for `p_θ ∈ H^α(R^K)`, satisfying the test on `|A|` directions bounds the
  expected discrepancy over all directions by `C(K,α) |A|^{-2α/(K-1)} · (Sobolev norm)`
  (spherical-harmonic / Marcinkiewicz-Zygmund interpolation). Large `α` (smooth DN
  embeddings, smooth Gaussian target) ⇒ `|A| = O(K)` suffices.
- **Resampling:** drawing fresh `A` each step accumulates coverage linearly in training time;
  `|A| = 16` resampled beats a fixed `|A|` of thousands.
- **Minibatch bias:** `E[|φ̂_n - ψ|^2] = |φ_θ - ψ|^2 + (1 - |φ_θ|^2)/n` (diagonal of the ECF
  double sum gives 1, off-diagonal `|φ_θ|^2`). So loss and gradient have explicit, vanishing
  `O(1/N)` bias — negligible even at `N = 16`.

## Implementation notes

- Integral by trapezoidal quadrature; **17 knots** suffice. Range `[-3,3]` or `[-5,5]` (not
  `[-1,1]`). Exploit evenness: integrate `[0, t_max]` with doubled weights and half-weighted
  endpoints. No complex arithmetic — `|φ̂ - φ|^2 = (cos-mean - φ)^2 + (sin-mean)^2`.
- `num_slices = |A|` ≈ 256 (16-2048 all work); directions Gaussian, normalized to unit norm,
  seeded by a step counter synchronized across GPUs (`all_reduce` MAX) so projections match.
- A degenerate test `T = mean(x)^2 + (std(x) - 1)^2` recovers VICReg in the limit (enforces
  `E[Z]=0`, `Cov(Z)=I`) — but being a 2-moment test it is non-identifiable; use Epps-Pulley.
- Computing the EP integral *exactly* (not by quadrature) recovers a kernel MMD per slice,
  but at `O(N^2)`; quadrature is what keeps it `O(N)`.

## Working code

Grounded in the canonical implementation (fast Epps-Pulley exploiting symmetry, sliced over
random unit directions, DDP via `all_reduce`).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def all_reduce_mean(x):
    """Global mean of a per-device batch statistic (identity on a single device)."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
    return x


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization with the Epps-Pulley test.

    Projects embeddings onto `num_slices` random unit directions (resampled each
    step) and on each projection measures the weighted-L2 distance between the
    empirical characteristic function and exp(-t^2/2), integrated by trapezoid.
    """

    def __init__(self, num_slices=256, lmbd=10.0, n_knots=17, t_max=3.0, sigma=1.0):
        super().__init__()
        self.num_slices = num_slices
        self.lmbd = lmbd
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        t = torch.linspace(0.0, t_max, n_knots)            # positive half-grid (use symmetry)
        dt = t_max / (n_knots - 1)
        weights = torch.full((n_knots,), 2 * dt)           # doubled for the negative half
        weights[[0, -1]] = dt                              # trapezoid endpoints
        window = torch.exp(-(t ** 2) / (2 * sigma ** 2))   # target CF == Gaussian window
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _epps_pulley(self, proj):
        # proj: [B, M] -- batch projected onto M directions
        B = proj.size(0)
        x_t = proj.unsqueeze(-1) * self.t                  # [B, M, n_knots]
        cos_mean = all_reduce_mean(x_t.cos().mean(0))      # Re phi_hat(t)
        sin_mean = all_reduce_mean(x_t.sin().mean(0))      # Im phi_hat(t)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        return (err @ self.weights) * (B * world)          # [M]

    def _directions(self, z):
        with torch.no_grad():                              # directions are not learned
            step = self.step.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(step, op=torch.distributed.ReduceOp.MAX)
            g = torch.Generator(device=z.device)
            g.manual_seed(int(step.item()))
            A = torch.randn(z.size(1), self.num_slices, device=z.device, generator=g)
            A = A / A.norm(p=2, dim=0)                      # unit-norm columns
        return A

    def forward(self, z):
        with torch.no_grad():
            A = self._directions(z)
            self.step += 1
        return self._epps_pulley(z @ A).mean()             # average over directions

    def pair(self, z1, z2):
        with torch.no_grad():
            A = self._directions(z1)                       # same synced step for both views
            self.step += 1
        return 0.5 * (
            self._epps_pulley(z1 @ A).mean() +
            self._epps_pulley(z2 @ A).mean()
        )


class CustomRegularizer(nn.Module):
    """Two-view anti-collapse regularizer: SIGReg on each view + invariance."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.sigreg = SIGReg(num_slices=num_slices, lmbd=lmbd)
        self.lmbd = lmbd

    def forward(self, z1, z2):
        sigreg = self.sigreg.pair(z1, z2)
        invariance_loss = F.mse_loss(z1, z2)
        return {
            "loss": invariance_loss + self.lmbd * sigreg,
            "sigreg": sigreg,
            "invariance_loss": invariance_loss,
        }
```

Multi-view prediction loss (for completeness; the invariance term above is its two-view
special case): with `V_g` global and `V` total views and per-sample center
`μ = mean of global-view embeddings`, `L_pred = (1/V) Σ_{v'} || μ - z_{v'} ||^2`, and the
full objective is `(1 - λ) L_pred + λ · SIGReg`.
