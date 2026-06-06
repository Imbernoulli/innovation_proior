# Mean–variance portfolio selection: the efficient frontier and the critical-line algorithm

## Problem

Given beliefs about a universe of `N` securities — a vector of expected returns `mu` and a
covariance matrix `Sigma` (symmetric, positive semidefinite, `Sigma_ij = rho_ij sigma_i sigma_j`)
— choose portfolio weights `w` (with `sum_i w_i = 1`, and `w_i >= 0` under long-only) that
trade off return against risk. Maximizing expected return alone is degenerate: `mu^T w` is linear
in `w`, so over the simplex it is maximized at a single vertex and never recommends diversification.

## Key idea

A portfolio's return `R = sum_i w_i R_i` has

- expected value `E = mu^T w` (linear), and
- variance `V = w^T Sigma w = sum_i w_i^2 sigma_ii + sum_i sum_{j != i} w_i w_j sigma_ij`.

The off-diagonal (covariance) terms are what matter: a security's contribution to portfolio risk
is its covariance with the rest of the holdings, not its standalone variance. Diversification
shrinks the diagonal but cannot eliminate the off-diagonal, so it reduces but does not abolish risk.

Because `E` and `V` trade off (the max-return and min-variance portfolios differ), the right object
is not a single optimum but the **efficient frontier**: the set of portfolios with minimum variance
for each level of expected return (equivalently maximum return for each level of variance). This
requires no utility function — it hands the investor the full menu of undominated trade-offs.

## The optimization

The frontier is the solution set of a parametric quadratic program. Three equivalent forms:

- Return-targeted: `min_w w^T Sigma w` s.t. `mu^T w >= m`, `1^T w = 1`, `l <= w <= u`, swept over `m`.
- Risk-targeted: `max_w mu^T w` s.t. `w^T Sigma w <= sigma_target^2`, `1^T w = 1`, swept over `sigma_target`.
- Lagrangian / quadratic utility: `max_w mu^T w - (delta/2) w^T Sigma w`, swept over risk-aversion
  `delta = 1/lambda` from `0` (max return) to `infinity` (min variance).

**Critical-line structure.** For `min (1/2) w^T Sigma w - lambda mu^T w` s.t. `1^T w = 1`, `l<=w<=u`,
the KKT stationarity is `Sigma w - lambda mu - gamma 1 - alpha + beta = 0` with complementary
slackness on the bound multipliers `alpha, beta`. Partition assets into the **free** set `F`
(`l_i < w_i < u_i`) and the **bounded** set `B` (pinned at a bound). On the free block,

```
w_F = Sigma_FF^{-1} ( lambda mu_F + gamma 1_F - Sigma_FB w_B ),
gamma = [ (1 - 1_B^T w_B) + 1_F^T Sigma_FF^{-1} Sigma_FB w_B
          - lambda (1_F^T Sigma_FF^{-1} mu_F) ] / ( 1_F^T Sigma_FF^{-1} 1_F ),
```

so `w_F` is **affine in `lambda`** — the *critical line*. The frontier is piecewise linear in `w`,
kinking only at **turning points** where an asset enters or leaves `F` (a free weight hits a bound,
or a bounded weight's multiplier changes sign). The **critical-line algorithm** starts at the
maximum-return corner (`lambda` large), walks `lambda` down to `0`, computes each turning point by a
linear-algebra update, and interpolates between them — yielding the exact frontier for arbitrary
bounds in a finite number of steps. Dropping the bounds recovers the classic global minimum-variance
portfolio `w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)` with variance `1 / (1^T Sigma^{-1} 1)`.

**Maximum-Sharpe / tangency portfolio.** Maximizing `(mu - r)^T w / sqrt(w^T Sigma w)` is not convex;
substitute `y = w/k` (`k > 0`), fix `(mu - r)^T y = 1`, minimize `y^T Sigma y` s.t. `1^T y = k`,
`k >= 0` (and box bounds scaled by `k`), then recover `w = y/k`. Within the critical-line frontier,
it is found by line-searching the Sharpe ratio along each turning-point segment.

## Code

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
