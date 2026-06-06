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

- Return-targeted: `min_w w^T Sigma w` s.t. `mu^T w = m`, `1^T w = 1`, `l <= w <= u`, swept over `m`.
- Risk-targeted: `max_w mu^T w` s.t. `w^T Sigma w <= V_target`, `1^T w = 1`, swept over `V_target`.
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


class EfficientFrontier:
    """Mean-variance optimal portfolios via a convex QP (long-only by default)."""

    def __init__(self, mu, Sigma, weight_bounds=(0, 1)):
        self.mu = np.asarray(mu).ravel()
        self.Sigma = np.asarray(Sigma)
        self.n = len(self.mu)
        self.w = cp.Variable(self.n)
        lo, hi = weight_bounds
        self._constraints = [cp.sum(self.w) == 1, self.w >= lo, self.w <= hi]

    def _solve(self, objective):
        cp.Problem(cp.Minimize(objective), self._constraints).solve()
        return self.w.value

    def min_volatility(self):
        return self._solve(cp.quad_form(self.w, self.Sigma, assume_PSD=True))

    def efficient_return(self, target):           # min variance for a target return
        self._constraints.append(self.w @ self.mu >= target)
        return self._solve(cp.quad_form(self.w, self.Sigma, assume_PSD=True))

    def efficient_risk(self, target_variance):    # max return for a risk budget
        self._constraints.append(
            cp.quad_form(self.w, self.Sigma, assume_PSD=True) <= target_variance)
        return self._solve(-(self.w @ self.mu))

    def max_quadratic_utility(self, delta=1.0):   # max mu^T w - (delta/2) w^T Sigma w
        util = (self.w @ self.mu
                - 0.5 * delta * cp.quad_form(self.w, self.Sigma, assume_PSD=True))
        return self._solve(-util)

    def max_sharpe(self, r=0.0):                   # tangency portfolio via y = w/k
        y, k = cp.Variable(self.n), cp.Variable()
        cons = [(self.mu - r) @ y == 1, cp.sum(y) == k, k >= 0,
                y >= 0.0 * k, y <= 1.0 * k]
        cp.Problem(cp.Minimize(cp.quad_form(y, self.Sigma, assume_PSD=True)),
                   cons).solve()
        return y.value / k.value


class CLA:
    """Critical-line algorithm: the entire efficient frontier, exactly."""

    def __init__(self, mu, Sigma, lB, uB):
        self.mean = np.asarray(mu, float).reshape(-1, 1)
        self.cov = np.asarray(Sigma, float)
        self.lB = np.asarray(lB, float).reshape(-1, 1)
        self.uB = np.asarray(uB, float).reshape(-1, 1)
        self.w, self.ls, self.g, self.f = [], [], [], []

    def _init_algo(self):
        # max-return corner: fill highest-mu assets to their upper bound
        idx = np.argsort(self.mean.ravel())
        w, i = np.copy(self.lB), len(idx)
        while w.sum() < 1:
            i -= 1
            w[idx[i]] = self.uB[idx[i]]
        w[idx[i]] += 1 - w.sum()
        return [idx[i]], w

    def _compute_w(self, covF_inv, covFB, meanF, wB):
        onesF = np.ones(meanF.shape)
        g1 = onesF.T @ covF_inv @ meanF
        g2 = onesF.T @ covF_inv @ onesF
        if wB is None:
            g, w1 = (-self.ls[-1] * g1 + 1) / g2, 0
        else:
            onesB = np.ones(wB.shape)
            w1 = covF_inv @ covFB @ wB
            g = (-self.ls[-1] * g1 + (1 - onesB.T @ wB + onesF.T @ w1)) / g2
        g = float(g)
        wF = -w1 + g * (covF_inv @ onesF) + self.ls[-1] * (covF_inv @ meanF)
        return wF, g

    def _compute_lambda(self, covF_inv, covFB, meanF, wB, i, bi):
        onesF = np.ones(meanF.shape)
        c1 = onesF.T @ covF_inv @ onesF
        c2 = covF_inv @ meanF
        c3 = onesF.T @ covF_inv @ meanF
        c4 = covF_inv @ onesF
        c = -c1 * c2[i] + c3 * c4[i]
        if c == 0:
            return None, bi
        if isinstance(bi, list):
            bi = bi[1] if c > 0 else bi[0]
        if wB is None:
            res = (c4[i] - c1 * bi) / c
        else:
            onesB = np.ones(wB.shape)
            l1 = onesB.T @ wB
            l3 = covF_inv @ covFB @ wB
            l2 = onesF.T @ l3
            res = ((1 - l1 + l2) * c4[i] - c1 * (bi + l3[i])) / c
        return float(res), bi

    # _solve() walks turning points: event (a) bound a free weight, event (b)
    # free a bounded weight; pick the binding lambda < current; append
    # (w, lambda, gamma, free-set); stop at lambda = 0; purge numerical /
    # dominated points. efficient_frontier() interpolates between turning points;
    # max_sharpe() line-searches the Sharpe ratio on each segment.
```
