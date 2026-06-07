Let me start from the continuous portfolio problem I already trust. If I only care about fractional weights, I can choose `x` on the simplex and trade return against covariance risk with a convex quadratic objective. In minimization form it is `0.5 * x' Sigma x - mu' x`, or the same thing with a risk-aversion multiplier in front of the variance. With ordinary lower and upper bounds this is a convex QP, so a CVXPY-style optimizer can solve it globally. The trouble is that the answer is a vector of real numbers, and the desk does not trade a vector of real numbers.

The first desk constraint is a support limit. I might be allowed to hold at most `K` names out of a large benchmark universe. The second is a buy-in floor: if I hold name `i` at all, I do not want a dust position; I want `x_i >= alpha_i`. The third is the actual deployment step: eventually the target has to become integer shares or integer lots. Rebalancing adds another layer, because I am usually moving from a current book `x0`, and large trades have impact. None of this changes why the covariance quadratic is attractive; it changes the feasible set around it.

The buy-in floor is already enough to break convexity. For one name the feasible set is `{0} union [alpha_i, u_i]`. If zero and `alpha_i` are both feasible, convexity would force every point between them to be feasible, but the whole interval `(0, alpha_i)` is exactly what I am trying to forbid. So no purely convex constraint in `x_i` can represent this floor-or-zero rule. The cardinality limit has the same structure in higher dimension. For each subset `S` with `|S| <= K`, the portfolios supported on `S` form a face of the simplex. The feasible set is the union of all those faces. If I average a feasible portfolio supported on `{1,2,3}` with another supported on `{4,5,6}`, I can get six positive weights, so the union is not convex.

A simple truncate-and-renormalize patch is tempting: solve the continuous QP, keep the `K` largest weights, and rescale. But that does not solve the constrained problem. Dropping one asset changes the covariance interactions among all the survivors, so the right weights on the remaining names are not the old weights multiplied by one scalar. A small position in the continuous optimum can be a hedge for a large one; removing it changes the marginal risk of the large one. The patch can also leave a survivor below `alpha_i` or above its cap. It is a heuristic, not the object I need.

An `L1` penalty does not rescue me either. In a long-only fully invested portfolio, `sum_i |x_i| = sum_i x_i = 1`, so the lasso-like penalty is a constant. Even outside the simplex, `L1` encourages shrinkage; it does not enforce "at most `K`" or "if positive, at least `alpha_i`." The requirement is not merely that weights become small. I need a yes-or-no decision for each name.

So I introduce that decision explicitly. Let `y_i` be binary, with `y_i = 1` meaning name `i` is in the portfolio. The linking constraints I want are

`alpha_i y_i <= x_i <= u_i y_i`.

The cases check out. If `y_i = 0`, then `0 <= x_i <= 0`, so the name is out. If `y_i = 1`, then `alpha_i <= x_i <= u_i`, so the name is held with its floor and cap. The cap `u_i` is also the tight big-M value; using a real cap rather than an arbitrary huge constant keeps the continuous relaxation tighter. Once the indicators exist, the holding count is just `sum_i y_i <= K`. I want `<= K`, not equality, because the rule is "at most"; forcing an extra name can waste a floor-sized position or even make an otherwise feasible choice infeasible.

That gives the clean mixed-integer quadratic model:

`minimize 0.5 * x' Q x + c' x`

subject to ordinary linear constraints, `sum_i y_i <= K`, `alpha_i y_i <= x_i <= u_i y_i`, and `y_i in {0,1}`. In the portfolio case without benchmark-relative terms, `Q` is the risk-aversion-scaled covariance and `c = -mu`. The objective remains convex in the continuous variables because `Q` is positive semidefinite; the only nonconvexity is integrality.

Now I need to be careful about the branch-and-bound story. If I relax `y_i in {0,1}` to `0 <= y_i <= 1`, I get a convex QP relaxation. Because I am minimizing, that relaxation gives a lower bound on any integer-feasible solution in the node. An incumbent integer solution gives an upper bound. If a node's lower bound is no better than the incumbent, I can prune the whole subtree. If a relaxed solution has a fractional `y_s`, I branch into `y_s = 0` and `y_s = 1`.

There is an older continuous-variable way to do the same search, and I must not overstate it. The inequality `sum_i x_i/u_i <= K` is only a surrogate relaxation of the support constraint, not an exact count. If at most `K` names are held and `x_i <= u_i`, then `sum_i x_i/u_i <= K` follows. The converse can fail: many small positive weights can have a small sum of `x_i/u_i` while still violating the support limit. That means the surrogate is useful for bounding, but it does not by itself enforce cardinality. The branch decisions have to restore the disjunction by pushing a chosen variable down to zero or up above its floor. A tailored implementation can branch directly on `x_s`: in the down branch, delete the variable or enforce `x_s = 0`; in the up branch, enforce `x_s >= alpha_s` and add `s` to the set of variables that are definitely in. Then each node solves the convex QP relaxation with the cardinality constraint removed and the already-branched-up lower bounds enforced. Pivoting methods such as Lemke's method are attractive there because a child node is only a small modification of its parent, so the parent basis can warm-start the next solve. The modeling version with binaries and a general MIQP solver expresses the same discrete choices more directly.

The portfolio objective itself needs one more pass. If I am tracking a benchmark `xB`, the risk term I actually care about can be benchmark-relative:

`0.5 * (x - xB)' Sigma (x - xB)`.

If I also pay symmetric quadratic impact from the current book, I add

`sum_i c_i (x_i - x0_i)^2`,

with `C = diag(c)`. I want the whole thing in the generic `0.5*x'Q*x + c'x` form. Expanding the benchmark term gives `0.5*x'Sigma*x - x' Sigma xB + 0.5*xB'Sigma*xB`, since `Sigma` is symmetric. Expanding impact gives `x'C x - 2*x0'C x + x0'C x0`, which is `0.5*x'(2C)x - (2C x0)'x + constant`. Including expected return `-r'x`, the nonconstant objective is

`0.5 * x' (Sigma + 2C) x - (r + Sigma xB + 2C x0)' x`.

If I use a risk-aversion multiplier `delta`, I replace `Sigma` by `delta Sigma` in the benchmark-risk part, so the linear benchmark term becomes `delta Sigma xB`. The constant is `0.5*xB'Sigma*xB + x0'C x0`, or the risk-scaled version of the first term; it does not affect the optimizer, so the code can drop it. This sign is important: benchmark risk contributes `+Sigma xB` to the effective return vector because the minimization objective contains `-x' Sigma xB`.

Sector-change constraints fit as linear rows. If sector `l` contains indices `S_l` and the allowed benchmark-relative change is `epsilon_l`, the absolute-value constraint

`|sum_{i in S_l} (x_i - xB_i)| <= epsilon_l`

is just the two inequalities `sum_{i in S_l} x_i <= epsilon_l + sum_{i in S_l} xB_i` and `-sum_{i in S_l} x_i <= epsilon_l - sum_{i in S_l} xB_i`. So they stay inside the convex-QP node relaxation.

For proportional transaction cost, I can split each trade into buy and sell variables, `x_i - x0_i = b_i - s_i`, `b_i >= 0`, `s_i >= 0`, and charge `tau_i^+ b_i + tau_i^- s_i`. For a fixed net trade, minimizing a positive cost drives the smaller of `b_i` and `s_i` to zero, so `b_i + s_i = |x_i - x0_i|` at optimum. In the symmetric implementation I do not need to write those variables by hand; CVXPY's `norm(x - x0, 1)` is the same convex proportional-cost term. Fixed ticket costs are different: a fee paid whenever the trade is nonzero is a step function, so an exact model needs another binary trade indicator. If ticket costs are second-order relative to impact or proportional cost, leaving them out keeps the core model quadratic and cleaner.

Round lots are not the same decision as support selection. I can solve the support-and-weight MIQP first, producing target weights, and then project those weights onto integer shares. The projection should not be naive rounding, because rounding each target dollar amount independently can overspend and can distort the allocation. Let `p_i` be the latest price, `T` the account value, and `x_i` the integer number of shares. The target dollar amount is `w_i T`, the deployed amount is `p_i x_i`, and the residual is `eta_i = w_i T - p_i x_i`. Introduce `u_i` so that `eta_i <= u_i` and `eta_i >= -u_i`; minimizing `sum_i u_i` squeezes `u_i` to `|eta_i|`. Add leftover cash `r = T - p'x` to the objective and require `x >= 0`, `r >= 0`. The result is a mixed-integer linear program:

`minimize sum_i u_i + r`.

That is exactly the integer allocation helper I want after the quadratic model chooses the target weights.

Now the code should look like the optimizer I already have: objective functions return CVXPY expressions for minimization, transaction cost is an `L1` term, and the allocation helper uses an integer variable with the `eta <= u`, `eta >= -u` absolute-value linearization.

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

I end up with a clean separation. The support decision is a mixed-integer convex quadratic model: binary `y` variables express the floor-or-zero disjunction and the count limit, while the expanded benchmark and impact terms keep the objective in `0.5*x'Q*x - r_tilde'x` form. Branch-and-bound gets its bounds from convex QP relaxations, whether the branching is written through binaries or directly on continuous variables. The final share list is a smaller integer linear projection that tracks the continuous target without overspending.
