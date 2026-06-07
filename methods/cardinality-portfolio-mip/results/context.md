## Research question

The continuous mean-variance rule is useful but incomplete for a trading desk. Given expected
returns `mu` and a positive semidefinite covariance matrix `Sigma`, the classical long-only
problem chooses weights `x` by minimizing a convex quadratic risk-return tradeoff under a
budget constraint and box bounds. The result is a fractional vector on the simplex.

The deployable problem is stricter. A manager may need at most `K` names, every held name may
need a buy-in floor `alpha_i`, industry-sector changes may need to stay near a benchmark, price
impact may depend on the size of the rebalance from the current book, and the final trade list
must become whole shares or round lots. A solution has to keep the covariance-aware quadratic
objective while expressing constraints of the form "zero or at least the floor" and "no more
than `K` nonzero holdings". Those are disjunctions, so the feasible set is no longer one convex
polytope.

## Background

Markowitz mean-variance optimization represents a portfolio by expected return `mu' x` and
variance `x' Sigma x`. With `sum_i x_i = 1` and ordinary lower/upper bounds, this is a convex
quadratic program. The same two-moment view underlies Roy's safety-first criterion and the
CAPM line of Sharpe and Lintner. This machinery works for continuously divisible weights.

The desk constraints change the geometry. The buy-in condition is the one-coordinate set
`{0} union [alpha_i, u_i]`; it has a gap between zero and `alpha_i`. The holding-count
condition is a union over all subsets of at most `K` names. Each fixed subset gives a convex
face of the simplex, but the union of many faces is not convex: averaging two feasible
portfolios supported on disjoint `K`-name sets can create a portfolio with more than `K`
nonzero names.

Operations research already has the modeling language for such disjunctions. A binary
indicator `y_i` can switch a continuous variable on and off through
`alpha_i y_i <= x_i <= u_i y_i`. If `y_i = 0`, the bounds force `x_i = 0`; if `y_i = 1`, they
restore the held interval. Summing the indicators gives a linear holding-count constraint.
The resulting model keeps the convex quadratic objective but adds integer variables.

Discrete trade size is a separate integer issue. If a lot of asset `i` costs `s_i p_i` and
the account value is `V`, holding `n_i` lots gives weight `n_i s_i p_i / V`; `n_i` is a
nonnegative integer. Solving in continuous weights and then naively rounding can overspend,
leave unused cash, or distort the target weights, so the allocation step is naturally an
integer linear projection under the budget.

Rebalancing also matters. If `x0` is the current portfolio, a symmetric quadratic impact model
adds `sum_i c_i (x_i - x0_i)^2` with `c_i > 0`. Linear transaction costs can be written with
buy and sell variables, or in the symmetric case as an `L1` norm of `x - x0`. Fixed ticket
costs are nonconvex step functions and need another trade indicator if they are modeled
exactly.

## Baselines

- **Continuous mean-variance QP (Markowitz 1952, 1959).** Minimize variance minus return, or
  maximize quadratic utility, over the simplex and ordinary bounds. It is convex and globally
  solvable. The gap is that the optimizer may spread weight over many names and does not know
  about buy-in floors, whole shares, sector turnover limits, or fixed ticket decisions.

- **Truncate and renormalize.** Solve the continuous problem, keep the `K` largest weights,
  zero the rest, and scale the survivors back to one. This is cheap, but it is not the
  constrained optimum: when one asset is dropped, the covariance interactions among the
  survivors change, so the survivors must be reoptimized. The truncation can also violate
  floors and caps.

- **Linear or piecewise-linear risk proxies.** Konno-Yamazaki-style mean absolute deviation
  and related approximations make integer linear programming easier. The gap is that the
  covariance quadratic is replaced by a proxy, so the diversification term that defined the
  mean-variance problem is no longer optimized directly.

- **Bienstock's tailored branch-and-bound for cardinality-bounded QP (1996).** The quadratic
  objective is kept, the support constraint is relaxed by the necessary surrogate
  `sum_i x_i/u_i <= K`, and the algorithm branches directly on continuous weights by forcing
  a variable down to zero or up above its lower bound. The surrogate is not an exact count:
  several small positive weights can satisfy it while exceeding the support limit. The branch
  decisions restore the combinatorial distinction. The gap is that this is a specialized
  solver design, not a general modeling scaffold for benchmark-relative sector moves,
  rebalancing impact, and round-lot deployment.

- **Metaheuristics for constrained frontiers.** Genetic algorithms, tabu search, simulated
  annealing, and local search encode subsets and search over them, often re-solving a
  continuous QP inside a fixed subset. They can produce useful feasible portfolios, but they
  do not provide the same global optimality certificate as branch-and-bound.

## Evaluation settings

- **Inputs.** Expected returns `r`; covariance `Sigma`; a benchmark `xB`; current holdings
  `x0`; per-name floors `alpha_i` and caps `u_i`; a maximum holding count `K`; sector sets
  `S_l` with change limits `epsilon_l`; impact coefficients `c_i`; latest prices `p_i`;
  lot sizes `s_i`; and account value `V`.

- **Feasible portfolios.** Long-only weights satisfy `sum_i x_i = 1`, each nonzero holding is
  at least `alpha_i`, at most `K` holdings are nonzero, caps are respected, and sector-level
  benchmark deviations obey `|sum_{i in S_l}(x_i - xB_i)| <= epsilon_l`.

- **Quantities to inspect.** Expected return `r' x`; benchmark-relative variance
  `(x - xB)' Sigma (x - xB)`; impact `sum_i c_i (x_i - x0_i)^2`; holding count; sector
  deviations; integer share counts; leftover cash; and dollar tracking error from the
  continuous target.

- **Scale.** Small universes make the discrete decisions inspectable. Larger universes require
  bounding and pruning, because the number of possible support sets grows combinatorially.

## Code framework

The available software pieces are a CVXPY-backed mean-variance optimizer and an integer
allocation helper. The missing pieces are the optimization rule that chooses the support and
the allocation rule that turns the target weights into shares.

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

    def _upper_bounds(self):
        lower, upper = self.weight_bounds
        return _as_vector(upper, self.n_assets, "upper bound")

    def solve(self, max_names, min_weight, benchmark_weights=None, current_weights=None,
              sector_map=None, sector_limits=None, impact=None,
              transaction_cost_rate=0.0, risk_aversion=1.0, solver=None):
        # TODO: choose at most max_names holdings, enforce the buy-in floor, add
        # benchmark-relative sector limits and rebalance costs, then solve the
        # quadratic model.
        raise NotImplementedError


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
        # TODO: choose integer share counts under the budget that track the
        # continuous target weights as closely as possible.
        raise NotImplementedError
```
