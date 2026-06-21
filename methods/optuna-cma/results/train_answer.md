We are handed a black box $f:\mathbb{R}^n\to\mathbb{R}$ to minimize, and the only operation available is to choose a point $x$ and read back $f(x)$. There is no gradient, no Hessian, no algebraic form, and every evaluation is a simulation or experiment that costs real money, so the one quantity worth spending and the one yardstick worth reporting is the number of function evaluations needed to reach a target. The landscapes that motivate this are the punishing ones: ill-conditioned, so the favorable directions are orders of magnitude narrower than the unfavorable ones; non-separable, so the problem does not decompose into $n$ independent one-dimensional searches; non-convex, rugged, and sometimes noisy, so anything that wants a clean derivative — BFGS, conjugate gradients — stalls or fails. I have ranks of sampled points and nothing else. On top of that I want two invariances treated as hard constraints rather than luxuries: the method should depend only on the *ordering* of the $f$-values (so it is unchanged by any strictly monotone rescaling of $f$), and it should be invariant to rotation of the search space (so performance proved on a separable problem transfers verbatim to any rotated, non-separable version of it).

The existing options each get part of this and miss the rest. Pure random search is trivially rank- and rotation-invariant but learns nothing, so on an ill-conditioned problem the fraction of useful samples shrinks like a volume ratio and the evaluation count explodes. The isotropic evolution strategy with Rechenberg's 1/5 success rule gives genuine step-size control but adapts only one scalar and assumes the distribution stays spherical, so on an anisotropic valley it must shrink to the *narrowest* good direction and crawl along the broad ones. Mutative self-adaptation of per-coordinate step sizes can fit axis-aligned anisotropy but privileges the coordinate axes, so a rotation of the problem destroys the benefit; correlated-mutation variants have the right expressiveness — a full oriented ellipsoid — but parameterize it with clumsy rotation angles and adapt all $(n^2+n)/2$ parameters through an indirect, noisy selection signal that needs impractically large populations. The selection signal is the deep disease here: in mutative control a strategy-parameter value is judged only through the single object point it happened to spawn, two values differ in selection probability by a small noisy margin, maximizing selection probability is not maximizing progress (it drives the step size systematically too small), and the realizable change rate per parameter falls as the number of parameters grows. Finally, the direct estimators — EDA, EMNA, cross-entropy — re-estimate the Gaussian from the selected points each generation, but they reference the *selected points' own mean*, which measures the within-cluster spread; on a slope that spread is narrower than the steps that produced it, so the variance collapses in exactly the direction the search should move, and the search converges prematurely, worst of all at the small populations an expensive $f$ forces. None of these achieves rank-only adaptation, rotation invariance, automatic learning of scale and orientation, reliable overall step control, and small-population efficiency all at once.

I propose CMA-ES, the $(\mu/\mu_w,\lambda)$ Covariance Matrix Adaptation Evolution Strategy. The search distribution is the maximum-entropy choice, a multivariate normal $N(m,\sigma^2 C)$: among all distributions with prescribed second moments it commits to nothing beyond what it has learned, and it singles out no coordinate. Writing the eigendecomposition $C = B D^2 B^\top$, with $B$ orthonormal (principal axes) and $D$ diagonal (axis lengths), I sample by drawing $z\sim N(0,I)$ and mapping $x = m + \sigma B D z$, so $B D z \sim N(0,C)$. The whole game is to manufacture, from one ranked generation, a good mean and a good ellipsoid. The ellipsoid is the prize because near a minimum the model is a convex quadratic $f(x)=\tfrac12(x-x^*)^\top H (x-x^*)$, and sampling with $C = H^{-1}$ turns its elliptical level sets into spheres — every direction equally good. So the ideal search covariance is the inverse Hessian up to scale, and learning $C$ from ranks is the black-box analogue of the inverse-Hessian preconditioner a quasi-Newton method builds from gradients.

What makes the method work, and the single hinge of the design, is the reference point in the covariance estimate. Sample $\lambda$ points, rank them, and consider the selected steps $y_{i:\lambda} = (x_{i:\lambda}-m)/\sigma$ in $\sigma$-units. The ordinary empirical covariance references the selected points' own mean and measures the spread *within* the surviving cluster; on a slope selection chops off the uphill tail, so this spread is smaller than what was sampled, and the estimator shrinks variance in the productive direction, geometrically, until the distribution collapses. Referencing instead the *old* distribution mean $m$ — the true center the points were drawn around — measures the spread of the *displacements* from where I was to where the good points are. These displacements point downhill and their outer products *grow* variance in the productive direction, so resampling tends to reproduce the successful step. I therefore estimate $C_\mu = \sum_i w_i\, y_{i:\lambda} y_{i:\lambda}^\top$ with decreasing weights $w_1\ge\dots\ge w_\mu>0$ summing to one. Weighting beyond flat truncation trusts better-ranked steps more, at the cost of averaging fewer effectively-independent samples; that cost is exactly the variance-effective selection mass $$\mu_{\mathrm{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2} = \frac{1}{\sum_i w_i^2},$$ which equals $\mu$ for flat weights and lies in $[1,\mu]$ in general. It is "how many independent samples' worth of information" the weighting actually uses, and it calibrates every learning rate, because every estimate formed from the selected steps has its variance set by $\mu_{\mathrm{eff}}$, not by $\mu$.

A single generation's $C_\mu$ is far too noisy at the small populations I want — to get its condition number on the sphere below ten needs $\mu_{\mathrm{eff}}$ of order $10n$ — so I do not trust one generation, I accumulate with exponential smoothing, which also forgets ancient generations sampled from a long-gone distribution. With rate $c_\mu$, $$C \leftarrow (1-c_\mu)\,C + c_\mu \sum_{i=1}^{\mu} w_i\, y_{i:\lambda} y_{i:\lambda}^\top,$$ the rank-$\mu$ update: a rank-$\min(\mu,n)$ correction each generation, decaying old $C$ toward new evidence over a backward horizon of $1/c_\mu$ generations. A stable rate is $c_\mu\sim\mu_{\mathrm{eff}}/n^2$ — more reliable evidence earns a faster rate, more parameters demand a slower one — and now small populations are an asset, because they give *more* generations per evaluation budget, and I need many cheap accumulating nudges rather than one expensive reliable estimate.

The rank-one term is the same idea folded in one step at a time, but the outer product $y y^\top = (-y)(-y)^\top$ is sign-blind, and the sign carries information: if successive mean-steps point the same way there is a genuine long axis to stretch along; if they alternate, the directional signal cancels. So I accumulate the *signed* steps into an evolution path before squaring. To keep the path a meaningful, unbiased length signal I demand that under random selection — where consecutive mean-steps are independent $N(0,C)/\sqrt{\mu_{\mathrm{eff}}}$ vectors — the path stays distributed $N(0,C)$. Solving the smoothing recursion $p_c \leftarrow a\,p_c + b\sqrt{\mu_{\mathrm{eff}}}\,(m_{\mathrm{new}}-m_{\mathrm{old}})/(c_m\sigma)$ for variance preservation forces $a^2+b^2=1$; with $a = 1-c_c$ the normalization is $b=\sqrt{c_c(2-c_c)}$, giving $$p_c \leftarrow (1-c_c)\,p_c + \sqrt{c_c(2-c_c)\,\mu_{\mathrm{eff}}}\;\frac{m_{\mathrm{new}}-m_{\mathrm{old}}}{c_m\sigma},\qquad C\leftarrow(1-c_1)\,C + c_1\, p_c p_c^\top.$$ The path amplifies persistent correlation: aligned steps sum to about $1/c_c$ times a single step, inflating the path length by roughly $1/\sqrt{c_c}$, so a single long axis (a cigar) is learned in $O(n)$ evaluations even though the raw rate $c_1\sim 2/n^2$ looks far too slow. Rank-$\mu$ shines when $\mu_{\mathrm{eff}}$ is large (rich within-generation evidence); rank-one shines when $\mu_{\mathrm{eff}}$ is small (the cross-generation path carries the load); there is no reason to choose, so I combine them, each with its own rate, the leading coefficient holding the total weight at one for stationarity: $$C \leftarrow \left(1 - c_1 - c_\mu\right)C + c_1\, p_c p_c^\top + c_\mu \sum_{i=1}^{\mu} w_i\, y_{i:\lambda} y_{i:\lambda}^\top.$$ The bottom-ranked samples also carry information about *bad* directions, so in the canonical active form the worst ranks take negative weights, each scaled by its Mahalanobis length $w_i^o = w_i\, n/(\|C^{-1/2}y_{i:\lambda}\|^2+\varepsilon)$ and the total negative mass bounded by $\alpha^-$, so bad directions shrink $C$ without a single long bad vector destroying positive definiteness; the leading coefficient then generalizes to $1 + c_1\delta_h - c_1 - c_\mu\sum_i w_i$.

The covariance adapts shape and per-direction scale, but it cannot move the overall step length at the rate the sphere demands (where the optimal $\sigma$ must shrink by a factor of order $\exp(1/4)$ every $n$ evaluations) — its largest reliable rate is $\sim\!\mu_{\mathrm{eff}}/n^2$, far too slow whenever $\mu_{\mathrm{eff}}\ll n$. So $\sigma$ gets its own fast controller, cumulative step-size adaptation, reading the scale off the realized steps. The idea: if successive steps point the same way the path is long and I am being inefficient (I could have covered the ground in fewer longer steps, so lengthen $\sigma$); if they cancel the path is short and I am overshooting (shorten $\sigma$); the neutral case is uncorrelated steps, which is what random selection gives. But the covariance path $p_c$ is $N(0,C)$, so its expected length depends on direction — judging by $\|p_c\|$ would confound alignment with "moving along a long axis of $C$." I therefore build a *conjugate* path, whitening each incoming step by $C^{-1/2}=B D^{-1} B^\top$ (rotate axes onto the coordinates, rescale every axis to unit length, rotate back), so that under random selection $p_\sigma\sim N(0,I)$ regardless of $C$ and its expected length is the constant $\chi_n = E\|N(0,I)\| = \sqrt{2}\,\Gamma(\tfrac{n+1}{2})/\Gamma(\tfrac{n}{2})$, evaluated by the closed-form asymptotic $\chi_n = \sqrt{n}\,(1 - \tfrac{1}{4n} + \tfrac{1}{21n^2}+\dots)$. The update is a multiplicative push of the path length toward $\chi_n$, $$p_\sigma \leftarrow (1-c_\sigma)\,p_\sigma + \sqrt{c_\sigma(2-c_\sigma)\,\mu_{\mathrm{eff}}}\;C^{-1/2}\langle y\rangle_w,\qquad \sigma \leftarrow \sigma\,\exp\!\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\|p_\sigma\|}{\chi_n}-1\right)\right),$$ with damping $d_\sigma$ bounding how fast a single scalar driving the whole scale can move, and the rule is unbiased in $\ln\sigma$ (so under random selection $\sigma$ is stationary on its natural multiplicative scale; the slight upward Jensen bias on noisy problems is the safe direction). One guard remains: a badly-initialized tiny $\sigma$ makes a near-linear run of aligned steps that would inject a spurious axis into $C$, so I stall the $p_c$ accumulation with a Heaviside switch $h_\sigma$ that fires only when $\|p_\sigma\|/\sqrt{1-(1-c_\sigma)^{2(g+1)}} < (1.4 + 2/(n+1))\chi_n$, the denominator de-biasing the early-generation transient of the exponential sum; when it stalls, $\delta_h = (1-h_\sigma)c_c(2-c_c)$ keeps the covariance accounting correct. The mean itself just moves to the weighted recombination $m \leftarrow m + c_m\sigma\langle y\rangle_w$ with $c_m=1$, the weighted average of the top $\mu\approx\lambda/2$ being the bias-variance sweet spot between the high-variance single best and the regressive full average.

The defaults all fall out of the structure rather than tuning: a logarithmically slow population $\lambda = 4+\lfloor 3\ln n\rfloor$ (small is good — more generations per budget), $\mu=\lfloor\lambda/2\rfloor$, log-decreasing raw weights $w'_i = \ln((\lambda+1)/2) - \ln i$, and every rate of order $1/n$ or $1/n^2$ scaled by $\mu_{\mathrm{eff}}$ and $n$ on the principle "fast enough to learn, slow enough to stay reliable." Because every use of $f$ is through its ranking and every update is written in coordinate-free geometric quantities ($y_{i:\lambda}$, the paths, $C$, and $C^{-1/2}$), the method inherits monotone-$f$ and rotation invariance by construction, and on a convex quadratic the selected steps systematically point along the low-$f$ directions, so $C$ converges to a multiple of $H^{-1}$ — quasi-Newton behavior on a black box, learned from ranks alone.

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
