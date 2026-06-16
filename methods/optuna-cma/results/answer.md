# CMA-ES — Covariance Matrix Adaptation Evolution Strategy

The `(μ/μ_w, λ)`-CMA-ES is a derivative-free, rank-based optimizer for non-linear,
non-convex, ill-conditioned, non-separable continuous problems `f: R^n -> R`. It samples a
population from a multivariate Gaussian `N(m, σ² C)`, ranks the points by `f`, and from the
ranking alone updates the mean `m`, the overall step size `σ`, and the full covariance `C`.
It uses only the *order* of the `f`-values, so it is invariant to monotone transformations of
`f`; its updates are coordinate-free, so rotations, reflections, and translations of the
search space do not change the algorithmic behavior. On a convex-quadratic `½(x−x*)ᵀH(x−x*)`,
the learned `C` converges to a multiple of `H^{-1}`, making it the black-box analogue of a
quasi-Newton method.

## Problem it solves

Minimize an expensive black-box `f` measured only in number of evaluations, on landscapes
where gradients are unavailable or useless (ruggedness, noise, non-convexity) and where the
favorable directions are anisotropic and off-axis. The challenge is to learn the conditioning
(scale and orientation of good directions) of `f` from ranks alone, reliably with few samples
per iteration.

## Key ideas

1. **Gaussian search distribution.** Sample `x = m + σ B D z`, `z ~ N(0,I)`, where
   `C = B D² Bᵀ` is the eigendecomposition. The covariance encodes the learned geometry; on a
   quadratic the ideal `C` is the inverse Hessian.
2. **Weighted recombination (move the mean).** `m ← m + c_m σ Σ_{i=1}^{μ} w_i y_{i:λ}` with
   `y_{i:λ} = (x_{i:λ} − m)/σ`, decreasing weights `w_1 ≥ … ≥ w_μ > 0`, `Σ w_i = 1`. The
   variance-effective selection mass `μ_eff = 1/Σ w_i²` ∈ [1, μ] calibrates every rate.
3. **Selected-steps covariance (the hinge).** Estimate the covariance of the *selected steps*
   `(x_{i:λ} − m)` referenced to the **old** mean `m` — not to the selected points' own mean —
   so the variance *grows* in the productive direction (rank-μ update) instead of collapsing
   (which is what referencing the new mean, à la EMNA/cross-entropy, does).
4. **Evolution paths (cumulation).** Accumulate signed mean-steps to recover the sign the
   outer product `y yᵀ` discards and to amplify persistent axes. An unwhitened path `p_c`
   drives a rank-one covariance update; a *conjugate* (whitened) path `p_σ`, whose length is
   direction-independent, drives the step size.
5. **Active covariance shrinking.** The worst-ranked directions may carry negative covariance
   weights, scaled by their Mahalanobis length, so bad directions shrink `C` without destroying
   positive definiteness.
6. **Cumulative step-size control (CSA).** Compare `‖p_σ‖` to its random-selection expectation
   `χ_n = E‖N(0,I)‖`; lengthen `σ` if steps are aligned (path long), shorten if they cancel
   (path short). Multiplicative, damped, unbiased on the log scale.

## Algorithm (one generation, g → g+1)

Sample, for `k = 1, …, λ`:

```
z_k ~ N(0, I)
y_k = B D z_k          ~ N(0, C)
x_k = m + σ y_k        ~ N(m, σ² C)
```

Rank `f(x_{1:λ}) ≤ … ≤ f(x_{λ:λ})`; set `y_{i:λ} = (x_{i:λ} − m)/σ`,
`⟨y⟩_w = Σ_{i=1}^{μ} w_i y_{i:λ}`.

Mean: `m ← m + c_m σ ⟨y⟩_w`.

Step size (conjugate path, whitened by `C^{-1/2} = B D^{-1} Bᵀ`):

```
p_σ ← (1 − c_σ) p_σ + sqrt(c_σ(2 − c_σ) μ_eff) · C^{-1/2} ⟨y⟩_w
σ   ← σ · exp( (c_σ / d_σ) ( ‖p_σ‖ / χ_n − 1 ) )
```

Covariance (rank-one via path + active rank-μ via population; Heaviside guard `h_σ`):

```
h_σ = 1 if ‖p_σ‖ / sqrt(1 − (1 − c_σ)^{2(g+1)}) < (1.4 + 2/(n+1)) χ_n  else  0
p_c ← (1 − c_c) p_c + h_σ sqrt(c_c(2 − c_c) μ_eff) ⟨y⟩_w
δ_h = (1 − h_σ) c_c (2 − c_c)
w_i^o = w_i                                            if w_i ≥ 0
        w_i n / (‖C^{-1/2} y_{i:λ}‖² + ε)              if w_i < 0
C   ← (1 + c_1 δ_h − c_1 − c_μ Σ_{i=1}^{λ} w_i) C
      + c_1 p_c p_cᵀ + c_μ Σ_{i=1}^{λ} w_i^o y_{i:λ} y_{i:λ}ᵀ
```

With positive weights only, `Σw = 1` and the covariance coefficient reduces to
`1 + c_1 δ_h − c_1 − c_μ`. The canonical implementation uses the active form above.

## Default parameters

```
λ   = 4 + ⌊3 ln n⌋,                                   μ = ⌊λ/2⌋
w'_i = ln((λ+1)/2) − ln i,                            i = 1,...,λ
μ_eff  = (Σ_{i≤μ} w'_i)² / Σ_{i≤μ} (w'_i)²
μ_eff^- = (Σ_{i>μ} w'_i)² / Σ_{i>μ} (w'_i)²
c_m  = 1
c_σ  = (μ_eff + 2) / (n + μ_eff + 5)
d_σ  = 1 + 2·max(0, sqrt((μ_eff − 1)/(n + 1)) − 1) + c_σ
c_c  = (4 + μ_eff/n) / (n + 4 + 2 μ_eff/n)
c_1  = 2 / ((n + 1.3)² + μ_eff)
c_μ  = min(1 − c_1 − 1e-8,  2 (μ_eff − 2 + 1/μ_eff) / ((n + 2)² + μ_eff))
α^-  = min(1 + c_1/c_μ,  1 + 2 μ_eff^-/(μ_eff + 2),  (1 − c_1 − c_μ)/(n c_μ))
w_i  = w'_i / Σ_{j: w'_j>0} w'_j                    if w'_i ≥ 0
       α^- w'_i / Σ_{j: w'_j<0} |w'_j|               if w'_i < 0
χ_n  = sqrt(n) (1 − 1/(4n) + 1/(21 n²))
```

Initialize `m` problem-dependent, `σ` ~ 0.3·(domain width), `C = I`, `p_c = p_σ = 0`, `g = 0`.

## Working code (ask/tell, faithful to the canonical `cmaes` implementation)

```python
import math
import numpy as np

_EPS = 1e-8
_SIGMA_MAX = 1e32


class CMA:
    """(mu/mu_w, lambda)-CMA-ES with an ask/tell interface. Adapts mean m,
    step size sigma, and covariance C of N(m, sigma^2 C) from ranked samples."""

    def __init__(self, mean, sigma, population_size=None, seed=None):
        n = len(mean)
        self.dim = n
        self.mean = np.asarray(mean, dtype=float).copy()
        self.sigma = float(sigma)
        self.C = np.eye(n)
        self.p_c = np.zeros(n)
        self.p_sigma = np.zeros(n)
        self.g = 0
        self.rng = np.random.RandomState(seed)
        self.B = None
        self.D = None

        self.population_size = population_size or 4 + math.floor(3 * math.log(n))
        self.mu = self.population_size // 2
        weights_prime = np.array([
            math.log((self.population_size + 1) / 2) - math.log(i + 1)
            for i in range(self.population_size)
        ])
        me = (np.sum(weights_prime[: self.mu]) ** 2) / np.sum(weights_prime[: self.mu] ** 2)
        me_minus = (np.sum(weights_prime[self.mu :]) ** 2) / np.sum(weights_prime[self.mu :] ** 2)
        self.mu_eff = me

        alpha_cov = 2.0
        self.c_1 = alpha_cov / ((n + 1.3) ** 2 + me)
        self.c_mu = min(
            1 - self.c_1 - 1e-8,
            alpha_cov * (me - 2 + 1 / me) / ((n + 2) ** 2 + alpha_cov * me / 2),
        )
        min_alpha = min(
            1 + self.c_1 / self.c_mu,
            1 + 2 * me_minus / (me + 2),
            (1 - self.c_1 - self.c_mu) / (n * self.c_mu),
        )
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        self.weights = np.where(
            weights_prime >= 0,
            weights_prime / positive_sum,
            min_alpha * weights_prime / negative_sum,
        )

        self.c_m = 1.0
        self.c_sigma = (me + 2) / (n + me + 5)
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.c_sigma
        self.c_c = (4 + me / n) / (n + 4 + 2 * me / n)
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    def _eigen_decomposition(self):
        if self.B is not None and self.D is not None:
            return self.B, self.D
        self.C = (self.C + self.C.T) / 2
        D2, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self.C = B @ np.diag(D ** 2) @ B.T
        self.B, self.D = B, D
        return B, D

    def ask(self):
        B, D = self._eigen_decomposition()
        z = self.rng.randn(self.dim)            # z ~ N(0, I)
        y = B @ (D * z)                          # y = B D z ~ N(0, C)
        return self.mean + self.sigma * y        # x ~ N(m, sigma^2 C)

    def tell(self, solutions):
        assert len(solutions) == self.population_size
        n = self.dim
        self.g += 1
        solutions = sorted(solutions, key=lambda s: s[1])   # rank by f, best first
        x = np.array([s[0] for s in solutions])
        y = (x - self.mean) / self.sigma                    # selected steps in sigma-units

        B, D = self._eigen_decomposition()
        self.B, self.D = None, None                         # C changes below; cache expires
        C_invsqrt = B @ np.diag(1.0 / D) @ B.T              # C^{-1/2} = B D^{-1} B^T

        # mean: weighted recombination (reference = old mean)
        y_w = np.sum(y[: self.mu].T * self.weights[: self.mu], axis=1)
        self.mean = self.mean + self.c_m * self.sigma * y_w

        # step-size: conjugate (whitened) path, length test vs chi_n
        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                        + math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
                        * (C_invsqrt @ y_w))
        norm_ps = np.linalg.norm(self.p_sigma)
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (norm_ps / self.chi_n - 1))
        self.sigma = min(self.sigma, _SIGMA_MAX)

        # covariance: sign-aware (unwhitened) path + Heaviside guard
        h_sigma = (norm_ps / math.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.g + 1)))
                   < (1.4 + 2 / (n + 1)) * self.chi_n)
        self.p_c = ((1 - self.c_c) * self.p_c
                    + h_sigma * math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w)
        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)

        w_io = self.weights * np.where(
            self.weights >= 0,
            1.0,
            n / (np.linalg.norm(C_invsqrt @ y.T, axis=0) ** 2 + _EPS),
        )

        rank_one = np.outer(self.p_c, self.p_c)
        rank_mu = np.einsum("i,ij,ik->jk", w_io, y, y)
        old_weight = 1 + self.c_1 * delta_h - self.c_1 - self.c_mu * np.sum(self.weights)
        self.C = (old_weight * self.C
                  + self.c_1 * rank_one
                  + self.c_mu * rank_mu)


# usage: a generic outer loop over an expensive black box
def optimize(f, opt, n_generations):
    for _ in range(n_generations):
        solutions = [(x, f(x)) for x in (opt.ask() for _ in range(opt.population_size))]
        opt.tell(solutions)
    return opt.mean
```

## Why each piece

- **Reference the old mean** in the covariance estimate → grows variance toward improvement
  (the selected-*steps* estimator), where the within-cluster estimator would shrink it.
- **Exponential smoothing (rank-μ)** at rate `c_μ ≈ μ_eff/n²` lets small populations win by
  accumulating many cheap generations rather than one expensive reliable estimate.
- **Evolution path (rank-one)** restores the step sign and amplifies persistent axes, learning
  a single long axis in `O(n)` despite `c_1 ≈ 2/n²`.
- **Active negative weights** use the bottom-ranked samples to remove variance from bad
  directions; their total mass and per-generation Mahalanobis length scaling keep `C` positive
  definite.
- **Whitened conjugate path** makes the step-length test direction-independent; comparing to
  `χ_n` lengthens `σ` when steps align, shortens it when they cancel; damping `d_σ` bounds the
  change rate; the rule is unbiased in `ln σ` (stationary under random selection).
- **All rates scale with `μ_eff` and `n`** — fast enough to learn, slow enough to stay
  reliable. Everything reads only ranks and is coordinate-free → monotone-`f` and rotation
  invariance, and `C → H^{-1}` on quadratics.
