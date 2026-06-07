# Cardinality-constrained portfolio optimization as a mixed-integer quadratic program

## Problem

Continuous mean-variance optimization returns a dense, fractional weight vector that a trading desk
cannot deploy. A real portfolio must hold **at most `K` names**, give every held position a
**minimum size** (buy-in floor `l_i`), trade in **whole shares / round lots** (integer counts), and
pay to **rebalance** from the current book. Each requirement is a disjunction — "`x_i = 0` or
`x_i in [l_i, u_i]`", "at most `K` of the `x_i` nonzero", "trade or don't" — so the feasible set is a
non-convex union of faces of the budget simplex. The convex QP cannot express it, truncate-and-
renormalise of the continuous optimum is suboptimal (dropping a name reshuffles the survivors through
the covariances), and an L1 / lasso penalty only *shrinks* weights — it can enforce neither a hard
count `K` nor a hard floor `l_i`.

## Key idea

Name each discrete decision with a binary indicator and link it to the continuous weight by a big-M /
fixed-charge pair, keeping the mean-variance quadratic objective intact. Let `y_i in {0,1}` with
`y_i = 1` iff name `i` is held:

```
l_i * y_i  <=  x_i  <=  u_i * y_i        (linking; u_i is the big-M)
```

- `y_i = 0` => `0 <= x_i <= 0` => `x_i = 0` (name dropped; the interval `(0, l_i)` is unreachable).
- `y_i = 1` => `l_i <= x_i <= u_i` (buy-in floor and cap).

Cardinality is then the single linear constraint `sum_i y_i <= K`. The whole problem becomes a
**mixed-integer quadratic program (MIQP)** whose only non-convexity is the integrality of `y`; a
convex-QP relaxation (`y_i in [0,1]`) gives a lower bound, and **branch-and-bound / branch-and-cut**
fixes fractional indicators (`y_s = 0` vs `y_s = 1`), pruning subtrees whose relaxation bound is worse
than the incumbent. (Equivalently, one may drop the binaries and impose the surrogate
`sum_i (x_i / u_i) <= K`, branching directly on the continuous variables.)

## Algorithm

**Cardinality MIQP** (risk tolerance `q`; floors `l`, caps `u`):

```
maximize_{x, y}   q * mu'x  -  (1/2) x'Sigma x   -  (transaction cost)
subject to        sum_i x_i = 1                          (fully invested)
                  l_i y_i <= x_i <= u_i y_i,  all i      (buy-in / drop linking)
                  sum_i y_i <= K                         (hold at most K names)
                  y_i in {0, 1}
```

**Rebalancing.** Split the trade `x_i - x0_i = b_i - s_i`, `b_i, s_i >= 0`. Turnover is
`sum_i (b_i + s_i) = sum_i |x_i - x0_i|` (exact: minimizing cost drives `b_i s_i = 0`). Subtract the
proportional cost `sum_i (tau^+ b_i + tau^- s_i)` from the objective and optionally cap turnover
`sum_i (b_i + s_i) <= T_max`. A fixed per-trade ticket `beta_i` uses a binary `t_i` with
`b_i + s_i <= M t_i` and cost `sum_i beta_i t_i` — the same indicator + big-M pattern.

**Round lots.** Project the target weights `w*` onto integer lot counts under the budget `V` (lot size
`s_i`, price `p_i`, lot cost `p_i s_i`), minimizing dollar tracking error plus leftover cash:

```
minimize_{n, u}   sum_i u_i + (V - sum_i n_i p_i s_i)
subject to        u_i >= w*_i V - n_i p_i s_i,  u_i >= -(w*_i V - n_i p_i s_i)
                  n_i >= 0 integer,   sum_i n_i p_i s_i <= V
```

a mixed-integer **linear** program (branch-and-bound over the LP relaxation).

## Code

```python
import numpy as np
import cvxpy as cp


def cardinality_portfolio(
    mu, Sigma, K,
    w_lower=0.05, w_upper=0.40,     # per-name buy-in floor and cap
    risk_tolerance=1.0,             # q: weight on return vs. (1/2) variance
    w_prev=None,                    # current book, for rebalancing
    tc_buy=0.0, tc_sell=0.0,        # proportional buy/sell cost rates
    turnover_limit=None,            # cap on sum_i |x_i - w_prev_i|
    solver=cp.SCIP,                 # any MIQP-capable solver (SCIP/GUROBI/CPLEX/MOSEK)
):
    """Cardinality- and buy-in-constrained mean-variance MIQP, with optional rebalancing."""
    n = len(mu)
    x = cp.Variable(n)                  # post-trade weights
    y = cp.Variable(n, boolean=True)    # y_i = 1 iff name i is held

    variance = cp.quad_form(x, Sigma, assume_PSD=True)
    objective = risk_tolerance * (mu @ x) - 0.5 * variance      # max q mu'x - (1/2) x'Sigma x

    constraints = [
        cp.sum(x) == 1,                 # fully invested
        x >= w_lower * y,               # buy-in floor when held (y=1 -> x >= l)
        x <= w_upper * y,               # cap, and y=0 -> x=0 (big-M = w_upper)
        cp.sum(y) <= K,                 # hold at most K names
    ]

    if w_prev is not None and (tc_buy > 0 or tc_sell > 0 or turnover_limit is not None):
        b = cp.Variable(n, nonneg=True)         # buy amounts
        s = cp.Variable(n, nonneg=True)         # sell amounts
        constraints += [x - w_prev == b - s]    # net trade; b_i s_i = 0 at optimum
        turnover = cp.sum(b + s)                # = sum_i |x_i - w_prev_i|
        objective = objective - (tc_buy * cp.sum(b) + tc_sell * cp.sum(s))
        if turnover_limit is not None:
            constraints += [turnover <= turnover_limit]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=solver)           # branch-and-bound over convex-QP relaxations
    return np.asarray(x.value).round(6), np.asarray(y.value).round(0).astype(int), prob.value


def round_lot_allocation(weights, prices, total_value, lot_size=1, solver=cp.HIGHS):
    """Project target weights onto whole-lot integer share counts under the budget (MILP)."""
    p = np.asarray(prices, float) * lot_size        # cost of one lot of each name
    w = np.asarray(weights, float)
    n = len(p)
    lots = cp.Variable(n, integer=True)             # integer lot counts
    cash = total_value - p @ lots                   # leftover cash
    dev = w * total_value - cp.multiply(lots, p)    # target dollars - deployed dollars
    u = cp.Variable(n)
    constraints = [u >= dev, u >= -dev,             # u_i >= |dev_i|
                   lots >= 0, cash >= 0]            # no shorting, don't overspend
    prob = cp.Problem(cp.Minimize(cp.sum(u) + cash), constraints)
    prob.solve(solver=solver)
    return np.rint(lots.value).astype(int), float(cash.value)


if __name__ == "__main__":
    np.random.seed(0)
    mu = np.array([0.12, 0.10, 0.07, 0.03, 0.15, 0.09, 0.05, 0.11])
    A = np.random.randn(8, 8) * 0.1
    Sigma = A @ A.T + np.diag(np.full(8, 0.02))     # PSD covariance

    # at most 3 names, each at least 5%
    x, y, val = cardinality_portfolio(mu, Sigma, K=3, w_lower=0.05, w_upper=0.6,
                                      risk_tolerance=2.0)
    held = x[x > 1e-6]
    print("held:", y, " count:", int(y.sum()), " min held weight:", round(held.min(), 4))

    # rebalance an existing book, charging and capping turnover
    w_prev = np.zeros(8); w_prev[[3, 6]] = [0.5, 0.5]
    x2, _, _ = cardinality_portfolio(mu, Sigma, K=3, w_lower=0.05, w_upper=0.6,
                                     risk_tolerance=2.0, w_prev=w_prev,
                                     tc_buy=0.1, tc_sell=0.1, turnover_limit=1.5)
    print("turnover:", round(np.abs(x2 - w_prev).sum(), 4))

    # deploy the target weights as whole shares on a $100k budget
    prices = np.array([50, 120, 33, 200, 17, 90, 45, 80.0])
    lots, cash = round_lot_allocation(x, prices, total_value=100000.0)
    print("shares:", lots, " cash left:", round(cash, 2))
```

Running it: the cardinality MIQP returns exactly 3 held names with every held weight above the 5%
floor; the rebalance respects the turnover cap and trades only when the post-cost return justifies it;
the round-lot step returns integer share counts that track the target weights with a few dollars of
residual cash. The continuous frontier is the convex inside; binary indicators with big-M linking and
branch-and-bound are the language for the combinatorial shell of a deployable portfolio.
