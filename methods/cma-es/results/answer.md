# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

## Problem

Minimize a black-box f: R^n → R accessible only by querying points and reading back values, with no gradient and no analytic form. The currency is the number of evaluations to reach a target f. The hard landscapes are ill-conditioned (curvature varies by orders of magnitude across directions) and non-separable (the productive directions are tilted off the coordinate axes), so the contours are long, thin, rotated ellipsoids. A method must progress efficiently on such ellipsoids while *learning* the local scaling and orientation from the evaluations alone, depending only on the ranking of f-values (invariance to monotonic transforms of f) and on nothing tied to the coordinate frame (invariance to rotation and scaling of the search space).

## Key idea

Sample from a multivariate normal N(m, σ²C). The optimal covariance for a convex-quadratic f(x)=½xᵀHx is the inverse Hessian H⁻¹: it whitens the tilted ellipsoid into a sphere, so adapting C toward H⁻¹ is a gradient-free, rank-only quasi-Newton that learns the metric. Adapt C by *directly* raising the likelihood of the mutation steps that selection just favored, referencing the *old* mean (so variance grows along productive directions rather than collapsing) — combining a rank-μ term (covariance of the selected steps, good when the population is large) with a rank-one term driven by an evolution path p_c that cumulates signed mean-displacements (good when the population is small, since it preserves the cross-generation correlation a squared step would erase). Carry the overall scale σ separately and on a faster timescale: cumulate a *conjugate* (whitened) path p_σ and compare its length to χ_n = E‖N(0,I)‖, the length pure chance would produce — longer means the steps are correlated and σ is too small (increase it), shorter means they oscillate and σ is too large (decrease it).

## Algorithm

Constants (defaults; each 1/c is a backward time horizon), with n the dimension:

- λ = 4 + ⌊3 ln n⌋,  μ = ⌊λ/2⌋.
- raw weights wᵢ' = ln((λ+1)/2) − ln i for i = 1..μ; weights wᵢ = wᵢ'/Σⱼ wⱼ' (so Σ wᵢ = 1).
- effective sample size μ_eff = (Σ wᵢ)² / Σ wᵢ² = 1/Σ wᵢ²  (in [1, μ]).
- c_σ = (μ_eff + 2)/(n + μ_eff + 5),  d_σ = 1 + 2·max(0, √((μ_eff−1)/(n+1)) − 1) + c_σ.
- c_c = (4 + μ_eff/n)/(n + 4 + 2μ_eff/n).
- c_1 = 2/((n+1.3)² + μ_eff),  c_μ = min(1 − c_1, 2(μ_eff − 2 + 1/μ_eff)/((n+2)² + μ_eff)).
- χ_n = √n (1 − 1/(4n) + 1/(21n²)).

State: m ∈ R^n, σ > 0, C = I (symmetric positive-definite), paths p_c = p_σ = 0.

Each generation (g counts generations from 0):

1. Factor C = B D² Bᵀ (eigendecomposition); sample x_k = m + σ B D z_k, z_k ~ N(0,I), for k = 1..λ. Equivalently x_k = m + σ y_k with y_k ~ N(0,C).
2. Evaluate f(x_k), sort to get the ranking x_{1:λ}, …, x_{λ:λ} (best first); let y_{i:λ} = (x_{i:λ} − m)/σ.
3. Mean: ⟨y⟩_w = Σ_{i=1}^μ wᵢ y_{i:λ};  m ← m + σ ⟨y⟩_w.
4. Conjugate path:  p_σ ← (1 − c_σ) p_σ + √(c_σ(2 − c_σ) μ_eff) · C^{−1/2} ⟨y⟩_w,  with C^{−1/2} = B D⁻¹ Bᵀ.
5. Heaviside stall:  h_σ = 1 if ‖p_σ‖ / √(1 − (1 − c_σ)^{2(g+1)}) < (1.4 + 2/(n+1)) χ_n, else 0.
6. Evolution path:  p_c ← (1 − c_c) p_c + h_σ √(c_c(2 − c_c) μ_eff) ⟨y⟩_w.
7. Covariance (δ(h_σ) = (1 − h_σ) c_c(2 − c_c) ≤ 1 corrects the variance lost when the path stalls):

   C ← (1 − c_1 − c_μ + c_1 δ(h_σ)) C + c_1 p_c p_cᵀ + c_μ Σ_{i=1}^μ wᵢ y_{i:λ} y_{i:λ}ᵀ.

8. Step-size:  σ ← σ · exp( (c_σ/d_σ)(‖p_σ‖/χ_n − 1) ).

Selection and all updates touch f only through the ranking, giving invariance to strictly monotonic transforms of f; adapting C toward the right affine encoding gives invariance to rotation and scaling of the search space.

## Code

```python
import numpy as np


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES for minimizing a black-box f: R^n -> R.

    Sample x = m + sigma * N(0, C); move the mean to the weighted best;
    learn C toward the inverse Hessian from the selected steps (rank-mu +
    a sign-preserving evolution path for rank-one); control sigma from the
    length of a whitened conjugate path.
    """

    def __init__(self, x0, sigma0):
        self.xmean = np.asarray(x0, dtype=float)
        self.sigma = float(sigma0)
        n = self.N = len(self.xmean)

        # --- strategy parameters (tuned defaults; each 1/c is a horizon) ---
        self.lam = 4 + int(3 * np.log(n))            # population size
        self.mu = self.lam // 2                       # number of parents

        w = np.log((self.lam + 1) / 2) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()                    # positive weights, sum 1
        self.mueff = 1.0 / np.sum(self.weights**2)    # effective sample size

        self.cs = (self.mueff + 2) / (n + self.mueff + 5)             # sigma path
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)  # C path
        self.c1 = 2 / ((n + 1.3)**2 + self.mueff)                     # rank-one rate
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff)
                         / ((n + 2)**2 + self.mueff))                  # rank-mu rate
        self.damps = (1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1)
                      + self.cs)                                       # sigma damping
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))   # E||N(0,I)||

        # --- dynamic state ---
        self.pc = np.zeros(n)        # evolution path for C  (carries the sign)
        self.ps = np.zeros(n)        # conjugate (whitened) path for sigma
        self.C = np.eye(n)           # covariance: learned metric ~ inverse Hessian
        self.counteval = 0

    def ask(self):
        # eigendecomposition C = B D^2 B^T; sample x = m + sigma * B D z, z~N(0,I)
        D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(D2)
        self.invsqrtC = (self.B * (1 / self.D)) @ self.B.T   # C^{-1/2} = B D^-1 B^T
        Z = np.random.randn(self.lam, self.N)
        self.Y = Z @ (self.B * self.D).T          # Y[k] ~ N(0, C)
        return self.xmean + self.sigma * self.Y   # X[k] ~ N(m, sigma^2 C)

    def tell(self, X, fitnesses):
        self.counteval += len(fitnesses)
        g = self.counteval / self.lam              # generation count
        order = np.argsort(fitnesses)              # rank by f -> rank invariance
        Y = self.Y[order]                          # selected steps, best first

        # move the mean: weighted average of the best mu steps
        ymean = self.weights @ Y[:self.mu]
        self.xmean = self.xmean + self.sigma * ymean

        # conjugate path for sigma: whiten the mean step with C^{-1/2}
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff)
                     * (self.invsqrtC @ ymean))

        # stall the C-path when ps is abnormally long (sigma far too small)
        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs)**(2 * g))
                < (1.4 + 2 / (self.N + 1)) * self.chiN)

        # evolution path for C: cumulate signed mean steps
        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean)

        # covariance update = rank-one (path) + rank-mu (selected steps)
        delta = (1 - hsig) * self.cc * (2 - self.cc)     # variance correction <= 1
        rank_mu = (Y[:self.mu] * self.weights[:, None]).T @ Y[:self.mu]
        self.C = ((1 - self.c1 - self.cmu + self.c1 * delta) * self.C
                  + self.c1 * np.outer(self.pc, self.pc)            # rank-one
                  + self.cmu * rank_mu)                            # rank-mu

        # step-size: compare path length to its random-selection expectation chiN
        self.sigma *= np.exp((self.cs / self.damps)
                             * (np.linalg.norm(self.ps) / self.chiN - 1))


def optimize(f, x0, sigma0, max_evals=None):
    es = CMAES(x0, sigma0)
    max_evals = max_evals or 1000 * es.N
    while es.counteval < max_evals:
        X = es.ask()
        es.tell(X, [f(x) for x in X])
    return es.xmean
```
