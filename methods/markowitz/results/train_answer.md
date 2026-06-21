An investor who already holds beliefs about a universe of $N$ securities — an expected return $\mu_i$ for each, and the full covariance structure $\Sigma_{ij} = \rho_{ij}\sigma_i\sigma_j$ — still faces the second-stage problem of turning those beliefs into an actual choice of weights $w$, with $\sum_i w_i = 1$ and $w_i \ge 0$ under long-only. The standard advice does not solve this. The present-value / dividend-discount tradition tells the investor to compute, for each security, a discounted anticipated return $R_i$, and to prefer the securities with the larger $R_i$, treating risk as a small bump in the discount rate that diversification washes away. But the portfolio's discounted return is $R = \sum_i w_i R_i$, a weighted average of fixed numbers with weights I get to choose on the simplex, and a weighted average is maximized by dumping everything onto the single largest $R_a$: set $w_a = 1$, zero elsewhere. This rule sends me to one corner and structurally can never prefer a diversified portfolio to the best undiversified one — no matter how the discount rates are formed. Since diversification is observed everywhere and is plainly sensible, the rule has to be rejected, and the defect is in the shape of the objective: a linear function over a simplex is always maximized at a vertex. The popular patch — spread across all the maximum-$\mu$ securities and invoke the law of large numbers so the realized yield converges to the expected one — fails for a different but related reason. The law of large numbers needs the averaged quantities to be roughly independent, but security returns are heavily intercorrelated; firms in the same industry have bad years together, so averaging them cancels nothing. The patch also quietly assumes a single portfolio is simultaneously maximum-mean and minimum-variance, which is false in general. And expected-utility maximization, though coherent, requires eliciting the investor's entire utility function $U(W)$, which no one knows how to do, and hands back no computable rule from the inputs an analyst actually has, namely $\mu$ and $\Sigma$. What is missing is a well-defined, computable rule that uses only first and second moments, makes diversification optimal for the right reason, and needs no utility function.

I propose mean–variance portfolio selection: characterize every portfolio by its two summary moments and choose among the resulting trade-offs along an efficient frontier, computed exactly by a critical-line algorithm. The starting point is to take variance seriously as its own object. The portfolio return $R = \sum_i w_i R_i$ has mean $E = \mu^T w$, which is linear, but its variance, grinding it out from the definition $V = E[(\sum_i w_i (R_i-\mu_i))^2]$, expands into
$$V = \sum_i\sum_j w_i w_j \sigma_{ij} = w^T \Sigma w = \sum_i w_i^2\,\sigma_{ii} + \sum_i\sum_{j\neq i} w_i w_j\,\sigma_{ij}.$$
The split is the crux. The diagonal piece, the own-variances weighted by $w_i^2$, does shrink as I spread wealth out — that is the part the law-of-large-numbers crowd implicitly treats as the whole story. But the off-diagonal piece, the double sum over every pair, grows like $N^2$ while the diagonal grows like $N$, so a well-spread portfolio's variance is dominated by the average covariance, a floor diversification can never push through unless the $\rho_{ij}$ vanish, which they do not. This also redefines the risk of a single security: the marginal effect $\partial V/\partial w_i = 2\sum_j \sigma_{ij} w_j$ shows a security's contribution to risk is its covariance with everything I already hold, not its standalone $\sigma_{ii}$. A volatile name that moves opposite the rest of the book can lower my variance. So the right diversification is across low-covariance names — across industries with different drivers — not merely across many names, and it falls straight out of which term of $V$ I am trying to shrink.

Now $E = \mu^T w$ is linear and $V = w^T \Sigma w$ is a convex quadratic ($\Sigma$ is PSD), and the two genuinely fight: the maximum-$E$ portfolio is an aggressive corner, the minimum-$V$ portfolio a balanced interior point, and they differ. There is no single optimum to name. What makes the method work is refusing to collapse the two numbers into one. Look at the whole set of attainable $(E,V)$ pairs and discard every dominated point: if portfolio $A$ has the same $E$ as $B$ at lower $V$, or the same $V$ at higher $E$, no one who likes return and dislikes risk picks $B$. What survives is the efficient frontier — minimum variance for each level of expected return, equivalently maximum return for each level of variance. Handing the investor that menu of undominated trade-offs sidesteps the utility function entirely; I need not know how they trade $E$ against $V$, only present all the non-dominated trades. (Roy, working in parallel, refuses utility the same way but commits to one point, the disaster-ratio maximizer of $(\mu_P - d)/\sigma_P$; same surface, a fixed choice on it.)

To compute the frontier I parametrize it with a single scalar by putting a Lagrange multiplier on return, minimizing over $w$
$$\tfrac{1}{2}\,w^T \Sigma w - \lambda\,\mu^T w \quad\text{s.t.}\quad \mathbf{1}^T w = 1,\ \ l \le w \le u,$$
where $\lambda = 0$ gives pure minimum variance (the bottom tip) and $\lambda \to \infty$ gives the maximum-return corner (the top); sweeping $\lambda$ upward walks the whole frontier. The KKT stationarity is $\Sigma w - \lambda\mu - \gamma\mathbf{1} - \alpha + \beta = 0$ with complementary slackness on the bound multipliers $\alpha,\beta$. Partition the assets into a free set $F$ (where $l_i < w_i < u_i$, so $\alpha_i=\beta_i=0$) and a bounded set $B$ (pinned at a known bound). On the free rows the bound multipliers drop, leaving $\Sigma_{FF}w_F + \Sigma_{FB}w_B = \lambda\mu_F + \gamma\mathbf{1}_F$, and since $\Sigma_{FF}$ is a principal block of a PD matrix it inverts:
$$w_F = \Sigma_{FF}^{-1}\big(\lambda\mu_F + \gamma\mathbf{1}_F - \Sigma_{FB}w_B\big),$$
$$\gamma = \frac{(1 - \mathbf{1}_B^T w_B) + \mathbf{1}_F^T\Sigma_{FF}^{-1}\Sigma_{FB}w_B - \lambda\big(\mathbf{1}_F^T\Sigma_{FF}^{-1}\mu_F\big)}{\mathbf{1}_F^T\Sigma_{FF}^{-1}\mathbf{1}_F},$$
where $\gamma$ is pinned by the budget $\mathbf{1}_F^T w_F = 1 - \mathbf{1}_B^T w_B$. Because $\gamma$ is itself affine in $\lambda$, substituting back makes $w_F(\lambda) = a_F + \lambda\,b_F$ purely affine in $\lambda$: this is the critical line in $N$ dimensions, with no geometry needed. (Dropping the bounds entirely recovers the textbook global minimum-variance portfolio $w^* = \Sigma^{-1}\mathbf{1}/(\mathbf{1}^T\Sigma^{-1}\mathbf{1})$ with variance $1/(\mathbf{1}^T\Sigma^{-1}\mathbf{1})$ — a reassuring special case at the bottom of the frontier.)

The frontier is therefore piecewise linear in $w$, and it kinks only at turning points where the free set changes — either a free weight slides into a bound, or a pinned weight's multiplier crosses zero and it wants to re-enter $F$. This is what makes the critical-line algorithm exact and finite. As $\lambda$ moves along a segment, $w_F(\lambda) = a_F + \lambda b_F$ slides linearly; for each free asset I solve the one linear equation $a_i + \lambda b_i = \text{boundary}$ to find the $\lambda$ at which it would hit its bound, and for each bounded asset I find the $\lambda$ at which its multiplier would change sign. The next turning point is the first such binding event. At it I move the offending asset between $F$ and $B$, re-form $\Sigma_{FF}^{-1}$, and continue. I start the walk at the unambiguous maximum-return corner — pour wealth into the highest-$\mu$ assets, filling each to its upper bound in order of decreasing $\mu$ until the budget is spent, the last partially-filled asset being the lone free one — then drive $\lambda$ down to zero, picking off turning points to the global minimum-variance portfolio. Between consecutive turning points the portfolio is the linear interpolation of the endpoints, so the whole frontier follows from a handful of linear-algebra updates. Two purges keep it honest: drop any stored point whose weights breach the budget or a box bound beyond a small tolerance (numerical hygiene against a near-singular $\Sigma$), and drop any dominated point so only the upper, efficient edge survives.

Finally, the single best risk-adjusted point — the tangency / maximum-Sharpe portfolio maximizing $(\mu - r)^T w / \sqrt{w^T \Sigma w}$ — is not a convex objective as written (an affine over a square root), so the general-solver route uses a change of variable: set $y = w/k$ with $k>0$, fix $(\mu - r)^T y = 1$, and minimize $y^T \Sigma y$ subject to $\mathbf{1}^T y = k$, $k \ge 0$ (box bounds scaled by $k$), then recover $w = y/k$ — a convex quadratic program. Inside the critical-line picture it is simpler still: the max-Sharpe point lies on the frontier, so on each turning-point segment $w(a) = a\,w_0 + (1-a)\,w_1$ the Sharpe ratio is a smooth function of the scalar $a$, line-searched (golden-section) per segment and maximized across segments. The result is two faithful readings of one object: a convex QP per frontier point, flexible about constraints, and the native critical-line walk that returns the entire frontier at once.

```python
import numpy as np
import cvxpy as cp


def _as_column(x):
    return np.asarray(x, dtype=float).reshape(-1, 1)


def _scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def _weight_bounds(n, weight_bounds):
    if len(weight_bounds) == n and not np.isscalar(weight_bounds[0]):
        bounds = np.asarray(weight_bounds, dtype=float)
        return bounds[:, 0], bounds[:, 1]
    lo, hi = weight_bounds
    lo = np.full(n, -1.0 if lo is None else lo) if np.isscalar(lo) or lo is None else np.asarray(lo, dtype=float)
    hi = np.full(n, 1.0 if hi is None else hi) if np.isscalar(hi) or hi is None else np.asarray(hi, dtype=float)
    return lo, hi


class EfficientFrontier:
    """Mean-variance optimal portfolios via a convex QP (long-only by default)."""

    def __init__(self, mu, Sigma, weight_bounds=(0, 1), solver=None):
        self.mu = np.asarray(mu).ravel()
        self.Sigma = np.asarray(Sigma)
        self.n = len(self.mu)
        self.w = cp.Variable(self.n)
        self.lo, self.hi = _weight_bounds(self.n, weight_bounds)
        self.solver = solver
        self._base_constraints = [self.w >= self.lo, self.w <= self.hi]

    def _solve(self, objective, constraints=()):
        constraints = list(self._base_constraints) + list(constraints)
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=self.solver)
        if self.w.value is None:
            raise ValueError(f"optimization failed: {prob.status}")
        return np.asarray(self.w.value).round(16) + 0.0

    def min_volatility(self):
        return self._solve(
            cp.quad_form(self.w, self.Sigma, assume_PSD=True),
            [cp.sum(self.w) == 1],
        )

    def efficient_return(self, target_return):
        return self._solve(
            cp.quad_form(self.w, self.Sigma, assume_PSD=True),
            [cp.sum(self.w) == 1, self.w @ self.mu >= target_return],
        )

    def efficient_risk(self, target_volatility):
        if target_volatility < 0:
            raise ValueError("target_volatility must be non-negative")
        variance = cp.quad_form(self.w, self.Sigma, assume_PSD=True)
        return self._solve(
            -(self.w @ self.mu),
            [cp.sum(self.w) == 1, variance <= target_volatility**2],
        )

    def max_quadratic_utility(self, risk_aversion=1.0):
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        util = (self.w @ self.mu
                - 0.5 * risk_aversion * cp.quad_form(self.w, self.Sigma, assume_PSD=True))
        return self._solve(-util, [cp.sum(self.w) == 1])

    def max_sharpe(self, risk_free_rate=0.0):
        if np.max(self.mu) <= risk_free_rate:
            raise ValueError("at least one expected return must exceed the reference rate")
        y, k = cp.Variable(self.n), cp.Variable()
        constraints = [
            (self.mu - risk_free_rate) @ y == 1,
            cp.sum(y) == k,
            k >= 0,
            y >= self.lo * k,
            y <= self.hi * k,
        ]
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, self.Sigma, assume_PSD=True)), constraints)
        prob.solve(solver=self.solver)
        if y.value is None or k.value is None:
            raise ValueError(f"optimization failed: {prob.status}")
        return (y.value / k.value).round(16) + 0.0


class CLA:
    """Critical-line algorithm: the entire efficient frontier, exactly."""

    def __init__(self, mu, Sigma, weight_bounds=(0, 1)):
        self.mean = np.asarray(mu, float).reshape(-1, 1)
        self.cov = np.asarray(Sigma, float)
        lo, hi = _weight_bounds(len(self.mean), weight_bounds)
        self.lB, self.uB = _as_column(lo), _as_column(hi)
        self.w, self.ls, self.g, self.f = [], [], [], []

    @staticmethod
    def _infnone(x):
        return float("-inf") if x is None else x

    def _init_algo(self):
        # max-return corner: fill highest-mu assets to their upper bound
        idx = np.argsort(self.mean.ravel())
        if float(np.sum(self.lB)) > 1 or float(np.sum(self.uB)) < 1:
            raise ValueError("bounds cannot satisfy a fully invested portfolio")
        w, i = np.copy(self.lB), len(idx)
        while float(np.sum(w)) < 1:
            i -= 1
            w[idx[i]] = self.uB[idx[i]]
        w[idx[i]] += 1 - float(np.sum(w))
        return [idx[i]], w

    def _compute_w(self, covF_inv, covFB, meanF, wB):
        onesF = np.ones(meanF.shape)
        g1 = _scalar(onesF.T @ covF_inv @ meanF)
        g2 = _scalar(onesF.T @ covF_inv @ onesF)
        if wB is None:
            g, w1 = (-self.ls[-1] * g1 + 1) / g2, 0
        else:
            onesB = np.ones(wB.shape)
            w1 = covF_inv @ covFB @ wB
            g = (-self.ls[-1] * g1 + (1 - _scalar(onesB.T @ wB) + _scalar(onesF.T @ w1))) / g2
        wF = -w1 + g * (covF_inv @ onesF) + self.ls[-1] * (covF_inv @ meanF)
        return wF, g

    def _compute_lambda(self, covF_inv, covFB, meanF, wB, i, bi):
        onesF = np.ones(meanF.shape)
        c1 = _scalar(onesF.T @ covF_inv @ onesF)
        c2 = covF_inv @ meanF
        c3 = _scalar(onesF.T @ covF_inv @ meanF)
        c4 = covF_inv @ onesF
        c = -c1 * _scalar(c2[i]) + c3 * _scalar(c4[i])
        if abs(c) < 1e-14:
            return None, None
        if isinstance(bi, list):
            bi = _scalar(bi[1]) if c > 0 else _scalar(bi[0])
        else:
            bi = _scalar(bi)
        if wB is None:
            res = (_scalar(c4[i]) - c1 * bi) / c
        else:
            onesB = np.ones(wB.shape)
            l1 = _scalar(onesB.T @ wB)
            l3 = covF_inv @ covFB @ wB
            l2 = _scalar(onesF.T @ l3)
            res = ((1 - l1 + l2) * _scalar(c4[i]) - c1 * (bi + _scalar(l3[i]))) / c
        return res, bi

    def _get_b(self, f):
        return [i for i in range(self.mean.shape[0]) if i not in f]

    @staticmethod
    def _reduce(matrix, rows, cols):
        if len(rows) == 0 or len(cols) == 0:
            return None
        return matrix[np.ix_(rows, cols)]

    def _get_matrices(self, f):
        b = self._get_b(f)
        return (
            self._reduce(self.cov, f, f),
            self._reduce(self.cov, f, b),
            self._reduce(self.mean, f, [0]),
            self._reduce(self.w[-1], b, [0]),
        )

    def _purge_num_err(self, tol=1e-9):
        i = 0
        while i < len(self.w):
            bad_budget = abs(float(np.sum(self.w[i])) - 1) > tol
            bad_bounds = np.any(self.w[i] < self.lB - tol) or np.any(self.w[i] > self.uB + tol)
            if bad_budget or bad_bounds:
                del self.w[i], self.ls[i], self.g[i], self.f[i]
            else:
                i += 1

    def _purge_excess(self):
        i, repeat = 0, False
        while True:
            if not repeat:
                i += 1
            if i == len(self.w) - 1:
                break
            mu_i = _scalar(self.w[i].T @ self.mean)
            repeat = False
            for j in range(i + 1, len(self.w)):
                if mu_i < _scalar(self.w[j].T @ self.mean):
                    del self.w[i], self.ls[i], self.g[i], self.f[i]
                    repeat = True
                    break

    def _solve(self):
        f, w = self._init_algo()
        self.w, self.ls, self.g, self.f = [np.copy(w)], [None], [None], [f[:]]
        while True:
            l_in = i_in = bi_in = None
            if len(f) > 1:
                covF, covFB, meanF, wB = self._get_matrices(f)
                covF_inv = np.linalg.inv(covF)
                for j, asset in enumerate(f):
                    lam, bi = self._compute_lambda(
                        covF_inv, covFB, meanF, wB, j, [self.lB[asset], self.uB[asset]]
                    )
                    if lam is not None and lam > CLA._infnone(l_in):
                        l_in, i_in, bi_in = lam, asset, bi

            l_out = i_out = None
            if len(f) < self.mean.shape[0]:
                for asset in self._get_b(f):
                    covF, covFB, meanF, wB = self._get_matrices(f + [asset])
                    covF_inv = np.linalg.inv(covF)
                    lam, _ = self._compute_lambda(
                        covF_inv, covFB, meanF, wB, meanF.shape[0] - 1, self.w[-1][asset]
                    )
                    if lam is not None and (self.ls[-1] is None or lam < self.ls[-1]):
                        if lam > CLA._infnone(l_out):
                            l_out, i_out = lam, asset

            if (l_in is None or l_in < 0) and (l_out is None or l_out < 0):
                self.ls.append(0)
                covF, covFB, meanF, wB = self._get_matrices(f)
                covF_inv = np.linalg.inv(covF)
                meanF = np.zeros(meanF.shape)
            else:
                if CLA._infnone(l_in) > CLA._infnone(l_out):
                    self.ls.append(l_in)
                    f.remove(i_in)
                    w[i_in] = bi_in
                else:
                    self.ls.append(l_out)
                    f.append(i_out)
                covF, covFB, meanF, wB = self._get_matrices(f)
                covF_inv = np.linalg.inv(covF)

            wF, gamma = self._compute_w(covF_inv, covFB, meanF, wB)
            for j, asset in enumerate(f):
                w[asset] = wF[j]
            self.w.append(np.copy(w))
            self.g.append(gamma)
            self.f.append(f[:])
            if self.ls[-1] == 0:
                break

        self._purge_num_err()
        self._purge_excess()

    def min_volatility(self):
        if not self.w:
            self._solve()
        variances = [_scalar(w.T @ self.cov @ w) for w in self.w]
        return self.w[int(np.argmin(variances))].ravel()

    def _golden_section(self, obj, a, b, args=(), minimum=True, tol=1e-9):
        sign = 1 if minimum else -1
        r, c = 0.618033989, 1 - 0.618033989
        x1, x2 = r * a + c * b, c * a + r * b
        f1, f2 = sign * obj(x1, *args), sign * obj(x2, *args)
        for _ in range(int(np.ceil(-2.078087 * np.log(tol / abs(b - a))))):
            if f1 > f2:
                a, x1, f1 = x1, x2, f2
                x2 = c * a + r * b
                f2 = sign * obj(x2, *args)
            else:
                b, x2, f2 = x2, x1, f1
                x1 = r * a + c * b
                f1 = sign * obj(x1, *args)
        return (x1, sign * f1) if f1 < f2 else (x2, sign * f2)

    def _eval_sr(self, a, w0, w1, risk_free_rate):
        w = a * w0 + (1 - a) * w1
        ret = _scalar(w.T @ self.mean) - risk_free_rate
        vol = _scalar(w.T @ self.cov @ w) ** 0.5
        return ret / vol

    def max_sharpe(self, risk_free_rate=0.0):
        if not self.w:
            self._solve()
        candidates, ratios = [], []
        for i in range(len(self.w) - 1):
            w0, w1 = np.copy(self.w[i]), np.copy(self.w[i + 1])
            a, ratio = self._golden_section(
                self._eval_sr, 0, 1, args=(w0, w1, risk_free_rate), minimum=False
            )
            candidates.append(a * w0 + (1 - a) * w1)
            ratios.append(ratio)
        return candidates[int(np.argmax(ratios))].ravel()

    def efficient_frontier(self, points=100):
        if not self.w:
            self._solve()
        mus, sigmas, weights = [], [], []
        per_segment = max(points // max(len(self.w), 1), 2)
        grid = np.linspace(0, 1, per_segment)[:-1]
        for i in range(len(self.w) - 1):
            if i == len(self.w) - 2:
                grid = np.linspace(0, 1, per_segment)
            for a in grid:
                w = self.w[i + 1] * a + (1 - a) * self.w[i]
                weights.append(np.copy(w).ravel())
                mus.append(_scalar(w.T @ self.mean))
                sigmas.append(_scalar(w.T @ self.cov @ w) ** 0.5)
        return mus, sigmas, weights
```
