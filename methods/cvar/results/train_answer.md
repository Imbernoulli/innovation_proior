The problem is to choose portfolio weights that make large losses rare and bounded. The standard regulatory number, Value-at-Risk, is the α-quantile of the loss distribution, but it is a poor objective for optimization. Because VaR is an order statistic of the scenario losses, it is non-convex and non-smooth in the weights; small changes in the portfolio can reorder scenarios and switch which one defines the quantile, producing many local minima. Worse, VaR ignores everything that happens beyond the quantile, so two portfolios with identical VaR can have very different tail severity, and it can fail subadditivity: pooling two independent risks can make VaR larger than the sum of the parts, which means it can punish diversification instead of rewarding it.

The right replacement is a tail measure that is convex in the portfolio, accounts for how bad the tail is beyond the threshold, and is coherent. The natural candidate is the mean of the α-tail, the average loss in the worst 1 − α of outcomes, which is known as Conditional Value-at-Risk (CVaR) or expected shortfall. Writing it as E[L | L ≥ VaR_α] is conceptually correct but computationally poisoned, because it is defined through VaR, the very non-convex quantity we want to avoid. The way to make CVaR usable is to introduce an auxiliary threshold variable ζ that is optimized jointly with the weights.

The key object is the Rockafellar-Uryasev auxiliary function F_α(w, ζ) = ζ + (1/(1 − α)) · E[(L(w, y) − ζ)^+], where (t)^+ = max(t, 0). For fixed w, this is convex in ζ, and minimizing it over ζ gives exactly the CVaR. The minimizer condition is Ψ(ζ^−) ≤ α ≤ Ψ(ζ), which is precisely the definition of ζ being an α-quantile; the lower endpoint of the minimizing interval is VaR_α. In the continuous case, substituting ζ* = VaR_α recovers E[L | L ≥ VaR_α]. With discrete scenario data, where there can be an atom of probability sitting exactly at the VaR threshold, the same formula automatically splits that atom correctly so that the tail weight stays exactly 1 − α, giving a coherent measure without any special case.

Because the loss is linear in the weights, L(w, y) = −R(y)^T w, the function F_α is jointly convex in (w, ζ). Therefore minimizing CVaR over the portfolio is the same as the joint convex problem min_{w ∈ X, ζ ∈ ℝ} F_α(w, ζ). There are no nested quantile computations and no local minima. On a finite panel of N equally-weighted return scenarios R_k, replace the expectation by the empirical average and lift each positive part to an epigraph variable u_k ≥ L_k(w) − ζ with u_k ≥ 0. Since each u_k has a positive coefficient in the objective, at the optimum it equals (L_k(w) − ζ)^+ exactly. With L_k(w) = −R_k^T w, all constraints become linear, turning the problem into a linear program in n + 1 + N variables.

```python
import cvxpy as cp
import numpy as np


class CVaREfficientPortfolio:
    """Mean-CVaR portfolio optimization via the auxiliary-variable LP.

        min_{w, zeta, u}  zeta + (1 / (N (1 - alpha))) * sum_k u_k
        s.t.  u_k >= L_k(w) - zeta,  u_k >= 0,  w in X,
        where L_k(w) = -R_k . w.

    The optimal value is alpha-CVaR; zeta is an alpha-quantile threshold.
    Data are returns (gains); loss is the negative of the return.
    """

    def __init__(self, expected_returns, returns, alpha=0.95, weight_bounds=(0, 1)):
        self.returns = np.asarray(returns)          # (N scenarios, n assets)
        self.expected_returns = np.asarray(expected_returns)
        self.N, self.n = self.returns.shape
        self.alpha = alpha                          # confidence level, e.g. 0.95
        self.lower, self.upper = weight_bounds
        self.w = cp.Variable(self.n)                # portfolio weights
        self.zeta = cp.Variable()                   # alpha-quantile threshold
        self.u = cp.Variable(self.N)                # loss-exceedance epigraph vars

    def _feasible_region(self, market_neutral=False):
        cons = [self.w >= self.lower, self.w <= self.upper]
        cons.append(cp.sum(self.w) == (0 if market_neutral else 1))
        return cons

    def _cvar_expr(self):
        return self.zeta + 1.0 / (self.N * (1 - self.alpha)) * cp.sum(self.u)

    def _cvar_constraints(self):
        # u_k >= (L_k - zeta) and u_k >= 0, with L_k = -R_k . w
        #   <=>  R_k . w + zeta + u_k >= 0  and  u_k >= 0
        return [self.u >= 0,
                self.returns @ self.w + self.zeta + self.u >= 0]

    def min_cvar(self, market_neutral=False):
        objective = self._cvar_expr()
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        cp.Problem(cp.Minimize(objective), constraints).solve()
        return self.w.value                          # self.zeta.value is an alpha-quantile

    def efficient_return(self, target_return, market_neutral=False):
        objective = self._cvar_expr()
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        constraints.append(self.expected_returns @ self.w >= target_return)
        cp.Problem(cp.Minimize(objective), constraints).solve()
        return self.w.value

    def efficient_risk(self, target_cvar, market_neutral=False):
        objective = self.expected_returns @ self.w
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        constraints.append(self._cvar_expr() <= target_cvar)
        cp.Problem(cp.Maximize(objective), constraints).solve()
        return self.w.value
```
