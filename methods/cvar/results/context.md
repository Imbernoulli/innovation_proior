# Context: Optimizing tail risk in a portfolio

## Research question

A portfolio is a vector of positions `w` chosen from a feasible set `X` (budget, no-short or box bounds, perhaps a target-return constraint). Its future loss `L(w, y) = -R(y)ᵀw` is a random variable driven by an uncertain market vector `y` (asset returns `R`). The goal is to choose `w` to control the right tail of the loss distribution — to make large losses unlikely and not too severe — rather than its average.

The standard tail number is **Value-at-Risk**. At confidence level `α ∈ (0,1)` (typically 0.90, 0.95, 0.99), `VaR_α(w)` is the smallest loss threshold `ζ` such that the loss stays at or below `ζ` with probability at least `α`:

```
VaR_α(w) = min{ ζ : P(L(w,y) ≤ ζ) ≥ α }   (the α-quantile of the loss).
```

VaR is written into banking regulation. The question is how to choose `w ∈ X` to control tail risk, working from scenario data, with a measure that can be carried through a solver at scale.

## Background

**The quantile as a function of the decision.** Consider scenario data: `N` equally likely sampled losses `L_1, …, L_N`. `VaR_α` is the `⌈αN⌉`-th order statistic. As `w` varies, the order statistic is a max/min over changing subsets — for linear scenario losses, a continuous piecewise-linear surface with kinks where the active scenario changes. Two `w`'s that put the same value at the `⌈αN⌉`-th position but very different values *above* it receive the same VaR.

**VaR and subadditivity — a worked construction.** Take two corporate bonds, each independently defaulting with probability `p = 0.04`, in which case it loses a fixed amount `L > 0` (otherwise it loses 0). Use confidence `α = 0.95`, so the tail mass is `1 − α = 0.05`.

- *One bond.* The loss is `L` with probability `0.04` and `0` with probability `0.96`. Since `P(loss ≤ 0) = 0.96 ≥ 0.95`, the 95%-quantile is `0`, so `VaR_0.95 = 0`.
- *Two independent bonds, A+B.* `P(both survive) = 0.96² = 0.9216`, `P(exactly one defaults) = 2·0.04·0.96 = 0.0768`, `P(both default) = 0.0016`. Now `P(loss ≤ 0) = 0.9216 < 0.95`, while `P(loss ≤ L) = 0.9216 + 0.0768 = 0.9984 ≥ 0.95`. Hence `VaR_0.95(A+B) = L`.

So `VaR(A+B) = L > 0 = VaR(A) + VaR(B)`: pooling two independent risks makes VaR larger than the sum of the parts.

**The coherence axioms.** Artzner, Delbaen, Eber and Heath (1999) ask which properties a risk measure `ρ` (a number assigned to a random outcome) should satisfy, and call a measure satisfying all four **coherent**:

- **Subadditivity (S):** `ρ(X₁ + X₂) ≤ ρ(X₁) + ρ(X₂)` — "a merger does not create extra risk"; diversification cannot hurt.
- **Positive homogeneity (PH):** `ρ(λX) = λ ρ(X)` for `λ ≥ 0` — scaling a position scales its risk.
- **Monotonicity (M):** if one position is always at least as good as another, its risk is no greater.
- **Translation invariance (T):** adding a sure amount of the reference (cash) instrument shifts the risk by exactly that amount.

They show quantile-based VaR violates (S), and they propose a coherent alternative — a *tail conditional expectation* / "worst conditional expectation" — characterized abstractly as a supremum of expectations over a family of "generalized scenarios" / probability measures.

**Mean shortfall / expected shortfall, defined through VaR.** The actuarial and extreme-value literature uses the *mean excess loss* / *mean shortfall* `E[ L | L > VaR_α ]` — the average loss given that VaR is breached. As written it is defined *through* `VaR_α(w)`: to evaluate it for a given `w` one first locates `VaR_α(w)`. For loss distributions with atoms (scenario/finite-sample models always have atoms), the conditional expectation `E[L | L > VaR]` places the probability mass sitting exactly at the VaR threshold on one side of the conditioning event.

**Scenarios and discreteness.** In practice the loss distribution is given by a finite Monte-Carlo or historical sample `y_1, …, y_N`, each with probability `1/N`. The loss CDF `Ψ(w, ζ) = P(L(w,y) ≤ ζ)` is then a step function with jumps at the sampled loss values: an order statistic in `w`, an atom at the threshold, and a distinction between "loss `>` VaR" and "loss `≥` VaR".

## Baselines

- **Markowitz mean-variance (1952).** The template for "risk measure inside an optimization." Risk is the return variance `σ²(w) = wᵀ Σ w`; one minimizes `wᵀ Σ w` subject to a return floor `μᵀ w ≥ R*` and budget `Σ wᵢ = 1`, a convex quadratic program with a clean efficient frontier. Variance penalizes upside and downside symmetrically and coincides with a tail measure when returns are elliptical/normal.

- **Parametric VaR (RiskMetrics, 1994–96).** Assume returns are jointly normal; then `VaR_α ∝ σ`, recovering mean-variance in disguise — a convex, tractable standard deviation.

- **Historical / Monte-Carlo simulation VaR.** Resample or simulate `N` scenarios, compute the loss in each, read off the `⌈αN⌉`-th order statistic. Handles nonlinear instruments and arbitrary distributions.

- **Axiomatic coherent measures (Artzner et al. 1999).** Establish the four axioms and that VaR violates subadditivity; propose tail/worst conditional expectation as coherent, characterized abstractly via generalized scenarios.

- **Mean shortfall / mean excess loss.** `E[ L | L > VaR_α ]` — the tail average, defined through `VaR_α`.

## Evaluation settings

The natural yardstick is **scenario-based portfolio optimization**: a universe of `n` instruments, a panel of `N` historical or simulated joint-return scenarios `R_k ∈ ℝⁿ` (each weight `1/N`), and a feasible set `X` (long-only `0 ≤ wᵢ ≤ 1` and budget `Σ wᵢ = 1`, optionally a target-return constraint `μᵀ w ≥ R*` or a market-neutral `Σ wᵢ = 0`). Confidence levels are the standard `α ∈ {0.90, 0.95, 0.99}`. The metrics are the realized tail-risk number (the tail-average loss at level `α`) and the portfolio's expected return, traced out as an efficient frontier; an index-replication task (track a benchmark while bounding tail underperformance) is a representative protocol. Out-of-sample behaviour is checked on a held-out window just after the in-sample period. These datasets, instruments, and metrics all pre-exist the method.

## Code framework

The primitives that already exist: a returns panel, an expected-return vector, a convex-optimization modelling layer (`cvxpy`) with a variable for the weights and a generic feasible region, and a solver. The tail-risk objective and its constraints are the empty slot.

```python
import cvxpy as cp
import numpy as np

class TailRiskEfficientPortfolio:
    """Choose weights w in X to control the right tail of the loss L = -R w,
    given a panel of return scenarios. Tail measure + constraints: TODO."""

    def __init__(self, expected_returns, returns, alpha=0.95, weight_bounds=(0, 1)):
        self.returns = np.asarray(returns)        # shape (N_scenarios, n_assets), gains
        self.expected_returns = np.asarray(expected_returns)
        self.N, self.n = self.returns.shape
        self.alpha = alpha                        # confidence level
        self.lower, self.upper = weight_bounds
        self.w = cp.Variable(self.n)              # portfolio weights

    def _feasible_region(self, market_neutral=False):
        cons = [self.w >= self.lower, self.w <= self.upper]
        cons.append(cp.sum(self.w) == (0 if market_neutral else 1))
        return cons

    def _tail_risk_expr(self):
        # TODO: the level-alpha tail-risk objective of the loss,
        # computed from the scenario panel.
        pass

    def _tail_risk_constraints(self):
        # TODO: any constraints the objective requires.
        pass

    def min_tail_risk(self, market_neutral=False):
        objective = self._tail_risk_expr()        # TODO
        constraints = self._feasible_region(market_neutral) + self._tail_risk_constraints()
        cp.Problem(cp.Minimize(objective), constraints).solve()
        return self.w.value

    def efficient_return(self, target_return, market_neutral=False):
        # minimise tail risk s.t. expected return >= target. TODO: same slot.
        pass

    def efficient_risk(self, target_risk, market_neutral=False):
        # maximise expected return s.t. tail risk <= target_risk. TODO: same slot.
        pass
```
