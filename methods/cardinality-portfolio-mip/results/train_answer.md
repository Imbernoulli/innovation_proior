The classical Markowitz mean-variance portfolio is convex and easy to solve, but it produces fractional weights that are not directly tradeable. When a desk needs to deploy capital, the real problem has several non-convex requirements: at most K names can be held, every held name must satisfy a minimum buy-in floor, sector deviations from a benchmark must stay inside tight bands, rebalancing costs should reflect the move from the current book, and the final execution list must be whole shares or round lots. Each of these rules breaks the continuous convex simplex. The buy-in floor alone creates a gap between zero and alpha_i, so the feasible set for one asset is the non-convex union {0} ∪ [alpha_i, u_i]. The cardinality rule is a union over all subsets of size at most K, which is also non-convex. Simple heuristics such as truncating the continuous solution to its K largest weights and renormalizing do not solve the constrained optimum, because removing one asset changes the covariance structure among the survivors and can violate floors or caps. L1 penalties are not useful either in a long-only fully invested portfolio, since the L1 norm is fixed by the budget constraint.

The right way to handle these disjunctions is to introduce an explicit binary decision for each asset. I propose cardinality-constrained portfolio mixed-integer quadratic programming. The model keeps the covariance quadratic objective exactly as in Markowitz and layers binary variables on top to enforce the discrete portfolio structure. Let x_i be the portfolio weight and y_i ∈ {0, 1} be an indicator that asset i is held. The linking constraints alpha_i y_i ≤ x_i ≤ u_i y_i force x_i to be zero when y_i is zero and to lie between the floor alpha_i and the cap u_i when y_i is one. The cardinality constraint sum_i y_i ≤ K limits the number of active positions. Because the objective remains a convex quadratic in the continuous weights and all other constraints are linear, the formulation is a mixed-integer convex quadratic program (MIQP). Its continuous relaxation is a convex QP that provides lower bounds for branch-and-bound, and any fractional y_i in a relaxation can be branched into an out branch y_i = 0 and an in branch y_i = 1.

For the objective, I use the expanded benchmark-relative and impact-aware form. If r is the vector of expected returns, Sigma is the covariance matrix, xB is the benchmark, x0 is the current portfolio, and C = diag(c) collects the impact coefficients, then minimizing 0.5 (x - xB)' Sigma (x - xB) + sum_i c_i (x_i - x0_i)^2 - r' x is equivalent, after dropping constants, to minimizing 0.5 x' (Sigma + 2C) x - (r + Sigma xB + 2C x0)' x. A risk-aversion multiplier scales the Sigma terms. Sector controls |sum_{i ∈ S_l} (x_i - xB_i)| ≤ epsilon_l become two linear inequalities and fit directly into the convex QP relaxation.

After the continuous MIQP selects target weights, the share-allocation step projects those weights onto integer shares without overspending. Let p_i be the price, T the account value, and n_i the integer share count. The target dollar amount is w_i T, the deployed amount is p_i n_i, and the residual is eta_i = w_i T - p_i n_i. Introducing auxiliary variables u_i with eta_i ≤ u_i and eta_i ≥ -u_i lets the MILP minimize sum_i u_i + r, where r = T - p' n is the leftover cash. This produces a budget-respecting integer execution list that tracks the continuous target as closely as possible.

```python
import collections

import cvxpy as cp
import numpy as np
import pandas as pd


def portfolio_return(w, expected_returns, negative=True):
    sign = -1 if negative else 1
    return sign * (w @ expected_returns)


def portfolio_variance(w, cov_matrix):
    return cp.quad_form(w, cov_matrix, assume_PSD=True)


def transaction_cost(w, w_prev, k=0.001):
    return k * cp.norm(w - w_prev, 1)


def _as_vector(value, n, name):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"{name} must be scalar or length {n}")
    return arr


class DeskPortfolioModel:
    def __init__(self, expected_returns, cov_matrix, tickers=None,
                 weight_bounds=(0, 1), solver=None):
        self.expected_returns = np.asarray(expected_returns, dtype=float).reshape(-1)
        self.cov_matrix = np.asarray(cov_matrix, dtype=float)
        self.n_assets = len(self.expected_returns)
        self.tickers = list(tickers) if tickers is not None else list(range(self.n_assets))
        self.weight_bounds = weight_bounds
        self.solver = solver
        if self.cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("covariance matrix does not match expected returns")

    def _upper_bounds(self):
        lower, upper = self.weight_bounds
        return _as_vector(upper, self.n_assets, "upper bound")

    def solve(self, max_names, min_weight, benchmark_weights=None, current_weights=None,
              sector_map=None, sector_limits=None, impact=None,
              transaction_cost_rate=0.0, risk_aversion=1.0, solver=None):
        n = self.n_assets
        w = cp.Variable(n)
        y = cp.Variable(n, boolean=True)

        floor = _as_vector(min_weight, n, "min_weight")
        cap = self._upper_bounds()
        linear = self.expected_returns.copy()
        quadratic = risk_aversion * self.cov_matrix.copy()

        if benchmark_weights is not None:
            benchmark = _as_vector(benchmark_weights, n, "benchmark_weights")
            linear = linear + risk_aversion * (self.cov_matrix @ benchmark)
        else:
            benchmark = None

        if impact is not None:
            if current_weights is None:
                raise ValueError("current_weights are required when impact is supplied")
            current = _as_vector(current_weights, n, "current_weights")
            impact_vec = _as_vector(impact, n, "impact")
            C = np.diag(impact_vec)
            quadratic = quadratic + 2 * C
            linear = linear + 2 * (C @ current)
        elif current_weights is not None:
            current = _as_vector(current_weights, n, "current_weights")
        else:
            current = None

        objective = 0.5 * cp.quad_form(w, quadratic, assume_PSD=True) - linear @ w
        if current is not None and transaction_cost_rate:
            objective += transaction_cost(w, current, k=transaction_cost_rate)

        constraints = [
            cp.sum(w) == 1,
            w >= cp.multiply(floor, y),
            w <= cp.multiply(cap, y),
            cp.sum(y) <= max_names,
        ]

        if sector_map and sector_limits:
            if benchmark is None:
                raise ValueError("benchmark_weights are required for sector limits")
            for sector, members in sector_map.items():
                idx = [self.tickers.index(t) if t in self.tickers else int(t) for t in members]
                shift = cp.sum(w[idx]) - float(np.sum(benchmark[idx]))
                limit = float(sector_limits[sector])
                constraints += [shift <= limit, -shift <= limit]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=solver or self.solver)
        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise ValueError(f"optimization failed with status {prob.status}")

        weights = collections.OrderedDict(
            (ticker, float(value)) for ticker, value in zip(self.tickers, w.value)
            if abs(value) > 1e-10
        )
        selected = collections.OrderedDict(
            (ticker, int(round(value))) for ticker, value in zip(self.tickers, y.value)
        )
        return weights, selected, prob.value


class ShareAllocator:
    def __init__(self, weights, latest_prices, total_portfolio_value=10000):
        if not isinstance(weights, dict):
            raise TypeError("weights should be a dictionary of {ticker: weight}")
        if not isinstance(latest_prices, pd.Series):
            raise TypeError("latest_prices should be a pandas Series")
        self.weights = list(weights.items())
        self.latest_prices = latest_prices
        self.total_portfolio_value = total_portfolio_value

    @staticmethod
    def _remove_zero_positions(allocation):
        return {k: v for k, v in allocation.items() if v != 0}

    def lp_portfolio(self, solver=None):
        p = self.latest_prices.values
        n = len(p)
        w = np.fromiter([weight for _, weight in self.weights], dtype=float)

        x = cp.Variable(n, integer=True)
        r = self.total_portfolio_value - p.T @ x
        eta = w * self.total_portfolio_value - cp.multiply(x, p)
        u = cp.Variable(n)

        constraints = [eta <= u, eta >= -u, x >= 0, r >= 0]
        objective = cp.sum(u) + r

        opt = cp.Problem(cp.Minimize(objective), constraints)
        opt.solve(solver=solver)
        if opt.status not in {"optimal", "optimal_inaccurate"}:
            raise ValueError("integer allocation failed")

        vals = np.rint(x.value).astype(int)
        allocation = collections.OrderedDict(
            zip([ticker for ticker, _ in self.weights], [int(v) for v in vals])
        )
        return self._remove_zero_positions(allocation), float(r.value)
```
