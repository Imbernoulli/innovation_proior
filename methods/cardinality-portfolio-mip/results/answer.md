# Cardinality-constrained portfolio MIQP

## Problem

A continuous mean-variance portfolio can be convex and globally solvable while still being
undeployable. The desk version needs a hard limit of at most `K` holdings, a minimum weight
`alpha_i` for every held name, optional benchmark-relative sector-move limits, rebalancing
costs from the current book, and integer shares at the end.

## Method

Introduce a binary selection variable `y_i` for each asset and link it to the continuous weight:

```
alpha_i y_i <= x_i <= u_i y_i
sum_i y_i <= K
y_i in {0, 1}
```

The two linking inequalities exactly encode `x_i = 0` or `alpha_i <= x_i <= u_i`. With a
positive semidefinite quadratic objective and linear constraints, the model is a mixed-integer
convex quadratic program. Its continuous relaxation gives a bound for branch-and-bound; a
fractional selection decision is split into an out branch and an in branch.

For a benchmark-relative and impact-aware portfolio, use

```
minimize  -r' x + 0.5 (x - xB)' Sigma (x - xB) + sum_i c_i (x_i - x0_i)^2
```

which has the same optimizer as

```
minimize  0.5 x' (Sigma + 2C) x - (r + Sigma xB + 2C x0)' x
```

after dropping constants, with `C = diag(c)`. A risk-aversion multiplier scales the `Sigma`
terms. Sector controls such as `|sum_{i in S_l}(x_i - xB_i)| <= epsilon_l` become two linear
inequalities.

After the target weights are chosen, integer deployment solves

```
minimize  sum_i u_i + r
subject to eta_i = w_i T - p_i n_i
           eta_i <= u_i,  eta_i >= -u_i
           n_i >= 0 integer,  r = T - p' n >= 0
```

This is the same integer-allocation structure used by PyPortfolioOpt's `lp_portfolio`.

## Code

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
