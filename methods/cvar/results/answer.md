# CVaR optimization — the auxiliary-variable linear program

## Problem

Choose portfolio weights `w ∈ X` to control the right tail of the loss `L(w, y) = −R(y)ᵀ w`. The regulatory tail number, Value-at-Risk `VaR_α(w)` (the α-quantile of the loss), is the wrong thing to *minimize*: as a function of `w` it is non-convex and multi-extremal (an order statistic of the scenarios), it is blind to how large losses get *beyond* the quantile, and it is not subadditive — pooling independent risks can make it grow, so it penalizes diversification and is not a coherent risk measure.

## Key idea

Replace VaR with **Conditional Value-at-Risk** `φ_α(w)` = the mean of the α-tail of the loss (the average loss in the worst `1 − α` of outcomes). In the continuous case this is the same as `E[L | L ≥ VaR_α(w)]`, but that expression is defined *through* the quantile. The useful move is the auxiliary function

```
F_α(w, ζ) = ζ + (1/(1−α)) · E[ (L(w,y) − ζ)⁺ ],     (t)⁺ = max{t, 0}.
```

**Minimization formula.** `F_α(w, ·)` is convex in `ζ`, and

```
φ_α(w) = min_ζ F_α(w, ζ),     argmin_ζ F_α(w, ζ) ∋ VaR_α(w).
```

So minimizing `F` over `ζ` computes CVaR directly and exposes the quantile interval; its lower endpoint is VaR. No quantile needs to be evaluated first.

*Proof sketch.* With `Ψ(ζ) = P(L ≤ ζ)`, the one-sided derivatives are
`∂⁺F_α/∂ζ = (Ψ(ζ) − α)/(1−α)` and `∂⁻F_α/∂ζ = (Ψ(ζ⁻) − α)/(1−α)`
(the right derivative sees `{L > ζ}` and the left derivative sees `{L ≥ ζ}`, hence the `Ψ(ζ)` versus `Ψ(ζ⁻)` split). By convexity the minimizers are the `ζ` with `∂⁻ ≤ 0 ≤ ∂⁺`, i.e. `Ψ(ζ⁻) ≤ α ≤ Ψ(ζ)` — exactly the α-quantiles, whose lower endpoint is `VaR_α`. In the continuous case, plugging the minimizing `ζ* = VaR_α` back in gives a tail event of probability `1 − α`, so `(1/(1−α))E[(L−ζ*)⁺] = E[L − ζ* | L ≥ ζ*]` and `F_α(w, ζ*) = E[L | L ≥ ζ*] = φ_α(w)`. For loss distributions with an atom at `ζ*` (always the case for scenario data), the same minimum value equals the α-tail mean with the atom *split*, `λ·VaR_α + (1−λ)·E[L | L > VaR_α]`, `λ = (Ψ(ζ*) − α)/(1−α)`, which keeps the tail weight exactly `1 − α` and makes `φ_α` coherent.

**Joint convexity and the optimization shortcut.** When `L(w,y)` is convex in `w` (in particular linear, `L = −Rᵀw`), `F_α(w, ζ)` is *jointly* convex in `(w, ζ)`. Therefore

```
min_{w ∈ X} φ_α(w) = min_{(w, ζ) ∈ X × ℝ} F_α(w, ζ),
```

a single convex program over the portfolio and the threshold together — no nested quantile, no local minima. `φ_α` inherits coherence (subadditivity + positive homogeneity from joint sublinearity; monotonicity and translation-equivariance from the `min_ζ` form), and `φ_α(w) ≥ VaR_α(w)` always, so minimizing CVaR also bounds VaR.

## Final algorithm (scenario LP)

Given `N` equally-weighted return scenarios `R_k`, the empirical objective is
`F̃_α(w, ζ) = ζ + (1/(N(1−α))) Σ_k (L_k(w) − ζ)⁺`, `L_k(w) = −R_kᵀw`. Lift each positive part to an epigraph variable `u_k`:

```
minimize_{w, ζ, u}   ζ + (1/(N(1−α))) Σ_{k=1}^N u_k
subject to           u_k ≥ L_k(w) − ζ,   u_k ≥ 0,    k = 1, …, N
                     w ∈ X.
```

At the optimum `u_k = (L_k(w) − ζ)⁺` (each `u_k` carries a positive objective coefficient, so it is pushed to the lower envelope of its two constraints). With `L_k = −R_kᵀw` linear, this is a **linear program** in `n + 1 + N` variables; the optimal objective is the α-CVaR, and the optimal `ζ` is an α-quantile threshold whose lower argmin endpoint is the α-VaR.

## Code

```python
import cvxpy as cp
import numpy as np


class CVaREfficientPortfolio:
    """Mean-CVaR optimization via the auxiliary-variable LP.

        min_{w, zeta, u}  zeta + (1 / (N (1 - alpha))) * sum_k u_k
        s.t.  u_k >= L_k(w) - zeta,  u_k >= 0,  w in X,     L_k(w) = -R_k . w.

    Optimal value = alpha-CVaR; zeta is an alpha-quantile threshold.
    Data are returns (gains); loss is the negative of the return.
    """

    def __init__(self, expected_returns, returns, alpha=0.95, weight_bounds=(0, 1)):
        self.returns = np.asarray(returns)            # (N scenarios, n assets)
        self.expected_returns = np.asarray(expected_returns)
        self.N, self.n = self.returns.shape
        self.alpha = alpha                            # confidence level (e.g. 0.95)
        self.lower, self.upper = weight_bounds
        self.w = cp.Variable(self.n)                  # portfolio weights
        self.zeta = cp.Variable()                     # alpha-quantile threshold
        self.u = cp.Variable(self.N)                  # loss-exceedance epigraph vars

    def _feasible_region(self, market_neutral=False):
        cons = [self.w >= self.lower, self.w <= self.upper]
        cons.append(cp.sum(self.w) == (0 if market_neutral else 1))
        return cons

    def _cvar_expr(self):
        # F_alpha = zeta + (1 / (N (1 - alpha))) * sum_k u_k
        return self.zeta + 1.0 / (self.N * (1 - self.alpha)) * cp.sum(self.u)

    def _cvar_constraints(self):
        # u_k >= (L_k - zeta) and u_k >= 0, with L_k = -R_k . w
        #   <=>  R_k . w + zeta + u_k >= 0  and  u_k >= 0
        return [self.u >= 0,
                self.returns @ self.w + self.zeta + self.u >= 0]

    def min_cvar(self, market_neutral=False):
        """Minimise portfolio CVaR."""
        objective = self._cvar_expr()
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        cp.Problem(cp.Minimize(objective), constraints).solve()
        return self.w.value                            # self.zeta.value is an alpha-quantile

    def efficient_return(self, target_return, market_neutral=False):
        """Minimise CVaR subject to a floor on expected return."""
        objective = self._cvar_expr()
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        constraints.append(self.expected_returns @ self.w >= target_return)
        cp.Problem(cp.Minimize(objective), constraints).solve()
        return self.w.value

    def efficient_risk(self, target_cvar, market_neutral=False):
        """Maximise expected return subject to a CVaR ceiling (frontier sweep)."""
        objective = self.expected_returns @ self.w
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        constraints.append(self._cvar_expr() <= target_cvar)
        cp.Problem(cp.Maximize(objective), constraints).solve()
        return self.w.value
```

The same auxiliary variables turn a CVaR *constraint* `φ_α(w) ≤ ω` into linear constraints `ζ + (1/(N(1−α)))Σ u_k ≤ ω`, `u_k ≥ L_k − ζ`, `u_k ≥ 0`, so CVaR can shape risk inside any other (e.g. tracking-error-minimizing) convex program.
