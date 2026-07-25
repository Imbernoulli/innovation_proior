We are asked to minimize a black-box function f: R^n → R that can only be queried point by point, with no gradients and no closed form. The only currency that matters is the number of evaluations needed to reach a target value. The hard cases are ill-conditioned and non-separable landscapes, where the contour lines form long, thin, tilted ellipsoids. An isotropic evolution strategy that samples a spherical cloud around the current mean fails on these problems: a single global step size must be small enough not to overshoot the steep directions, which makes progress along the flat directions glacial, and a sphere has no preferred orientation, so it cannot align with rotated productive directions. Axis-parallel step sizes handle coordinate-wise scaling but collapse as soon as the problem is rotated. Mutative self-adaptation of the full distribution is indirect and noisy, forcing the population size to scale with the number of strategy parameters. Methods that fit a Gaussian to the selected points around their own sample mean, such as EMNA or the continuous cross-entropy method, systematically shrink variance along the gradient and converge prematurely. What is needed is a way to learn the full local geometry directly from the successful steps, using the old mean as the reference so that variance grows in the directions that are actually productive.

The method is the Covariance Matrix Adaptation Evolution Strategy, or CMA-ES. It maintains a multivariate normal search distribution N(m, σ^2 C). On a convex quadratic f(x)=½xᵀHx, the ideal covariance is C = H⁻¹, because that linear change of variables turns the tilted ellipsoid back into a sphere. CMA-ES therefore drives C toward the inverse local Hessian without ever seeing a gradient, using only the ranking of function values. Each generation it samples λ offspring, ranks them by f, and moves the mean to a weighted average of the best μ points. The covariance is updated from two complementary sources: a rank-μ term that accumulates the outer products of the selected steps around the old mean, and a rank-one term driven by an evolution path p_c that cumulates signed mean displacements. The rank-μ term works well when the population is large enough to give a reliable per-generation estimate; the rank-one term preserves the sign and correlation of consecutive steps, which is exactly the information a single outer product discards, and lets small populations learn dominant directions quickly. The overall scale σ is controlled separately on a faster timescale by a conjugate evolution path p_σ. This path is whitened by C^{-1/2} so its expected length is direction-independent; if the recent steps are correlated and the path is long, σ is increased, while if the steps cancel and the path is short, σ is decreased. Because all updates depend only on ranks, the algorithm is invariant to monotonic transforms of f, and because C learns the right affine encoding it is invariant to rotation and scaling of the search space.

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

By separating the learning of shape in C from the learning of scale in σ, and by using rank-μ and rank-one updates both referenced to the old mean, CMA-ES turns the problem of searching a tilted ellipsoid into the problem of searching a sphere after an affine transformation it learns on the fly. The invariance properties mean that performance on one representative problem carries over to any rotated or rescaled version, which is exactly what makes it effective on ill-conditioned, non-separable black-box objectives.
