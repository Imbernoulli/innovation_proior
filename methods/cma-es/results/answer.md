# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

## Problem

Minimize a black-box f: R^n → R accessible only by querying points and reading back values, with no gradient and no analytic form. The currency is the number of evaluations to reach a target f. The hard landscapes are ill-conditioned (curvature varies by orders of magnitude across directions) and non-separable (the productive directions are tilted off the coordinate axes), so the contours are long, thin, rotated ellipsoids. A method must progress efficiently on such ellipsoids while *learning* the local scaling and orientation from the evaluations alone, depending only on the ranking of f-values (invariance to monotonic transforms of f) and on nothing tied to the coordinate frame (invariance to rotation and scaling of the search space).

## Key idea

Sample from a multivariate normal N(m, σ²C). The optimal covariance for a convex-quadratic f(x)=½xᵀHx is the inverse Hessian H⁻¹: it whitens the tilted ellipsoid into a sphere, so adapting C toward H⁻¹ is a gradient-free, rank-only quasi-Newton that learns the metric. Adapt C by *directly* raising the likelihood of the mutation steps that selection just favored, referencing the *old* mean (so variance grows along productive directions rather than collapsing) — combining a rank-μ term (covariance of the selected steps, good when the population is large) with a rank-one term driven by an evolution path p_c that cumulates signed mean-displacements (good when the population is small, since it preserves the cross-generation correlation a squared step would erase). Carry the overall scale σ separately and on a faster timescale: cumulate a *conjugate* (whitened) path p_σ and compare its squared length to E‖N(0,I)‖² = n — longer means the steps are correlated and σ is too small (increase it), shorter means they oscillate and σ is too large (decrease it).

## Algorithm

Constants (defaults; each 1/c is a backward time horizon), with n the dimension:

- λ = 4 + ⌊3 ln n⌋,  μ = ⌊λ/2⌋.
- raw weights wᵢ' = ln((λ+1)/2) − ln i for i = 1..μ; weights wᵢ = wᵢ'/Σⱼ wⱼ' (so Σ wᵢ = 1).
- effective sample size μ_eff = (Σ wᵢ)² / Σ wᵢ² = 1/Σ wᵢ²  (in [1, μ]).
- c_σ = (μ_eff + 2)/(n + μ_eff + 5),  d_σ = 2μ_eff/λ + 0.3 + c_σ.
- c_c = (4 + μ_eff/n)/(n + 4 + 2μ_eff/n).
- c_1 = 2/((n+1.3)² + μ_eff),  c_μ = min(1 − c_1, 2(μ_eff − 2 + 1/μ_eff)/((n+2)² + μ_eff)).
- χ_n = √n (1 − 1/(4n) + 1/(21n²)) for the equivalent length-based step-size form.

State: m ∈ R^n, σ > 0, C = I (symmetric positive-definite), paths p_c = p_σ = 0.

Each generation (g counts generations from 0):

1. Factor C = B D² Bᵀ (eigendecomposition); sample x_k = m + σ B D z_k, z_k ~ N(0,I), for k = 1..λ. Equivalently x_k = m + σ y_k with y_k ~ N(0,C).
2. Evaluate f(x_k), sort to get the ranking x_{1:λ}, …, x_{λ:λ} (best first); let y_{i:λ} = (x_{i:λ} − m)/σ.
3. Mean: ⟨y⟩_w = Σ_{i=1}^μ wᵢ y_{i:λ};  m ← m + σ ⟨y⟩_w.
4. Conjugate path:  p_σ ← (1 − c_σ) p_σ + √(c_σ(2 − c_σ) μ_eff) · C^{−1/2} ⟨y⟩_w,  with C^{−1/2} = B D⁻¹ Bᵀ.
5. Heaviside stall:  h_σ = 1 if (‖p_σ‖²/n) / (1 − (1 − c_σ)^{2(g+1)}) < 2 + 4/(n+1), else 0.
6. Evolution path:  p_c ← (1 − c_c) p_c + h_σ √(c_c(2 − c_c) μ_eff) ⟨y⟩_w.
7. Covariance (δ(h_σ) = (1 − h_σ) c_c(2 − c_c) ≤ 1 corrects the variance lost when the path stalls):

   C ← (1 − c_1 − c_μ + c_1 δ(h_σ)) C + c_1 p_c p_cᵀ + c_μ Σ_{i=1}^μ wᵢ y_{i:λ} y_{i:λ}ᵀ.

8. Step-size:  σ ← σ · exp(min(1, (c_σ/(2d_σ))(‖p_σ‖²/n − 1))).

Selection and all updates touch f only through the ranking, giving invariance to strictly monotonic transforms of f; adapting C toward the right affine encoding gives invariance to rotation and scaling of the search space.

## Code

```python
import numpy as np


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES for minimizing a black-box f: R^n -> R."""

    def __init__(self, x0, sigma0, max_evals=None, target=None):
        self.xmean = np.asarray(x0, dtype=float)
        self.sigma = float(sigma0)
        n = self.N = len(self.xmean)
        self.max_evals = (int(max_evals) if max_evals is not None
                          else int(100 * (4 + int(3 * np.log(n)))
                                   + 150 * (n + 3)**2 * np.sqrt(4 + int(3 * np.log(n)))))
        self.target = target

        # --- strategy parameters (tuned defaults; all are ~1/timescale) ---
        self.lam = 4 + int(3 * np.log(n))            # population size
        self.mu = self.lam // 2                       # number of parents

        weights = np.zeros(self.lam)
        weights[:self.mu] = np.log(self.lam / 2 + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights[:self.mu] /= np.sum(weights[:self.mu])      # positive weights sum to 1
        self.weights = weights
        self.mueff = 1.0 / np.sum(weights[:self.mu]**2)     # effective sample size

        self.cs = (self.mueff + 2) / (n + self.mueff + 5)             # sigma path
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)  # C path
        self.c1 = 2 / ((n + 1.3)**2 + self.mueff)                     # rank-one rate
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff)
                         / ((n + 2)**2 + self.mueff))                  # rank-mu rate
        self.damps = 2 * self.mueff / self.lam + 0.3 + self.cs        # sigma damping

        # --- dynamic state ---
        self.pc = np.zeros(n)        # evolution path for C  (carries the sign)
        self.ps = np.zeros(n)        # conjugate (whitened) path for sigma
        self.C = np.eye(n)           # covariance: learned metric ~ inverse Hessian
        self.counteval = 0
        self.fitvals = np.array([])
        self.best_x = self.xmean.copy()
        self.best_f = np.inf

    def ask(self):
        # eigendecomposition C = B D^2 B^T; sample x = m + sigma * B D z, z~N(0,I)
        D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(D2)
        self.invsqrtC = (self.B * (1 / self.D)) @ self.B.T   # C^{-1/2} = B D^-1 B^T
        Z = np.random.randn(self.lam, self.N)
        Y = Z @ (self.B * self.D).T               # Y[k] ~ N(0, C)
        return self.xmean + self.sigma * Y        # X[k] ~ N(m, sigma^2 C)

    def tell(self, X, fitnesses):
        self.counteval += len(fitnesses)
        order = np.argsort(fitnesses)              # rank by f -> rank invariance
        X = np.asarray(X, dtype=float)[order]
        self.fitvals = np.asarray(fitnesses, dtype=float)[order]
        if self.fitvals[0] < self.best_f:
            self.best_f = float(self.fitvals[0])
            self.best_x = X[0].copy()
        xold = self.xmean.copy()

        # move the mean: weighted average of the best mu steps
        self.xmean = self.weights[:self.mu] @ X[:self.mu]
        ymean = (self.xmean - xold) / self.sigma
        Y = (X - xold) / self.sigma                # selected steps around the old mean

        # conjugate path for sigma: whiten the mean step with C^{-1/2}
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff)
                     * (self.invsqrtC @ ymean))

        # stall the C-path when ps is abnormally long (sigma far too small)
        ps2 = float(self.ps @ self.ps)
        hsig = ((ps2 / self.N)
                / (1 - (1 - self.cs)**(2 * self.counteval / self.lam))
                < 2 + 4 / (self.N + 1))

        # evolution path for C: cumulate signed mean steps
        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean)

        # covariance update = rank-one (path) + rank-mu (selected steps)
        c1a = self.c1 * (1 - (1 - float(hsig)**2) * self.cc * (2 - self.cc))
        rank_mu = (Y * self.weights[:, None]).T @ Y
        self.C = ((1 - c1a - self.cmu * np.sum(self.weights)) * self.C
                  + self.c1 * np.outer(self.pc, self.pc)   # rank-one
                  + self.cmu * rank_mu)                    # rank-mu

        # step-size: squared path length is compared to E||N(0,I)||^2 = N
        exponent = min(1.0, (self.cs / self.damps) * (ps2 / self.N - 1) / 2)
        self.sigma *= np.exp(exponent)

    def stop(self):
        reasons = {}
        if self.counteval >= self.max_evals:
            reasons["max_evals"] = self.max_evals
        if self.target is not None and self.best_f <= self.target:
            reasons["target"] = self.target
        if self.fitvals.size and self.fitvals[-1] - self.fitvals[0] < 1e-12:
            reasons["tolfun"] = 1e-12
        if np.linalg.cond(self.C) > 1e14:
            reasons["condition"] = np.linalg.cond(self.C)
        if self.sigma * np.sqrt(np.max(np.diag(self.C))) < 1e-11:
            reasons["tolx"] = 1e-11
        return reasons


def optimize(f, x0, sigma0, max_evals=None, target=None):
    es = CMAES(x0, sigma0, max_evals=max_evals, target=target)
    while not es.stop():
        X = es.ask()
        es.tell(X, [f(x) for x in X])
    return es.best_x if es.best_f < f(es.xmean) else es.xmean
```
