# Context: Optimizing tail risk in a portfolio

## Research question

A portfolio is a vector of positions `w` chosen from a feasible set `X` (budget, no-short or box bounds, perhaps a target-return constraint). Its future loss `L(w, y) = -R(y)ᵀw` is a random variable driven by an uncertain market vector `y` (asset returns `R`). The goal is to choose `w` to make *large losses* unlikely and not too severe — to control the right tail of the loss distribution rather than its average.

The standard tail number is **Value-at-Risk**. At confidence level `α ∈ (0,1)` (typically 0.90, 0.95, 0.99), `VaR_α(w)` is the smallest loss threshold `ζ` such that the loss stays at or below `ζ` with probability at least `α`:

```
VaR_α(w) = min{ ζ : P(L(w,y) ≤ ζ) ≥ α }   (the α-quantile of the loss).
```

VaR is written into banking regulation, so it is the natural target. The problem is that **minimizing VaR over `w` is the wrong optimization problem**, for two reasons that are facts about quantiles, knowable before any new measure is built:

1. **As a function of `w`, VaR is non-convex and can be discontinuous, with many local minima.** A quantile of a `w`-dependent distribution moves in jumps as `w` changes the ordering of scenarios; gradient/convex methods land on local optima that are not global. For scenario-based (finite-sample) loss models the VaR-in-`w` surface is piecewise constant with cliffs.
2. **VaR ignores the magnitude of losses beyond the threshold and is not subadditive.** Two portfolios with identical VaR can have wildly different tails past `ζ` — a thin overshoot vs. a fat catastrophe. And combining positions can make VaR *increase*, contradicting the intuition that diversification reduces risk.

What a usable solution must achieve: a tail-risk measure that (i) is **convex in `w`** so it can be minimized to a global optimum, (ii) **accounts for how bad the tail is** beyond the quantile, (iii) is **coherent** in the axiomatic sense below, and (iv) reduces, for scenario data, to something a solver can handle at large scale — ideally a **linear program**.

## Background

**The quantile is non-convex in the decision and blind beyond itself.** Consider scenario data: `N` equally likely sampled losses `L_1, …, L_N`. `VaR_α` is the `⌈αN⌉`-th order statistic. As `w` varies, the order statistic is a max/min over changing subsets — a non-convex, piecewise-constant function with kinks and jumps. Minimizing it is a combinatorial, multi-extremal problem. Worse, two `w`'s that put the same value at the `⌈αN⌉`-th position but very different values *above* it are scored identically: VaR cannot tell "lose a little more than `ζ`" from "lose everything."

**VaR fails subadditivity — a worked construction.** Take two corporate bonds, each independently defaulting with probability `p = 0.04`, in which case it loses a fixed amount `L > 0` (otherwise it loses 0). Use confidence `α = 0.95`, so the tail mass is `1 − α = 0.05`.

- *One bond.* The loss is `L` with probability `0.04` and `0` with probability `0.96`. Since `P(loss ≤ 0) = 0.96 ≥ 0.95`, the 95%-quantile is `0`, so `VaR_0.95 = 0`.
- *Two independent bonds, A+B.* `P(both survive) = 0.96² = 0.9216`, `P(exactly one defaults) = 2·0.04·0.96 = 0.0768`, `P(both default) = 0.0016`. Now `P(loss ≤ 0) = 0.9216 < 0.95`, while `P(loss ≤ L) = 0.9216 + 0.0768 = 0.9984 ≥ 0.95`. Hence `VaR_0.95(A+B) = L`.

So `VaR(A+B) = L > 0 = VaR(A) + VaR(B)`: pooling two independent risks makes VaR *larger* than the sum of the parts. A measure that punishes diversification is structurally unfit for a risk-budgeting framework where one wants to split a capital limit across desks and have the parts add up.

**The coherence axioms.** Artzner, Delbaen, Eber and Heath (1999) ask which properties a risk measure `ρ` (a number assigned to a random outcome) *should* satisfy, and call a measure satisfying all four **coherent**:

- **Subadditivity (S):** `ρ(X₁ + X₂) ≤ ρ(X₁) + ρ(X₂)` — "a merger does not create extra risk"; diversification cannot hurt.
- **Positive homogeneity (PH):** `ρ(λX) = λ ρ(X)` for `λ ≥ 0` — scaling a position scales its risk.
- **Monotonicity (M):** if one position is always at least as good as another, its risk is no greater.
- **Translation invariance (T):** adding a sure amount of the reference (cash) instrument shifts the risk by exactly that amount.

They prove quantile-based VaR violates (S), and they propose a coherent alternative — a *tail conditional expectation* / "worst conditional expectation" — but characterize it abstractly (as a supremum of expectations over a family of "generalized scenarios" / probability measures). That object is coherent in principle but, in their own assessment, "impractical... can only be calculated in very narrow circumstances": there is no general recipe for *minimizing* it over a portfolio `w`.

**Mean shortfall / expected shortfall, defined through VaR.** The actuarial and extreme-value literature already uses the *mean excess loss* / *mean shortfall* `E[ L | L > VaR_α ]` — the average loss given that VaR is breached. Conceptually this is exactly the tail-severity number one wants: it looks past the quantile and averages the bad tail. But as written it is defined *through* `VaR_α(w)`: to evaluate it for a given `w` you must first locate `VaR_α(w)`, the very object that is non-convex and ill-behaved in `w`. Using it as an optimization objective inherits VaR's pathology. For loss distributions with atoms (and scenario/finite-sample models always have atoms), the naive conditional expectation `E[L | L > VaR]` also turns out not to be coherent, because the probability mass sitting exactly at the VaR threshold is handled inconsistently.

**Why scenarios force discreteness.** In practice the loss distribution is given by a finite Monte-Carlo or historical sample `y_1, …, y_N`, each with probability `1/N`. The loss CDF `Ψ(w, ζ) = P(L(w,y) ≤ ζ)` is then a step function with jumps at the sampled loss values. Every difficulty above — the cliffs in VaR, the atom at the threshold, the gap between "loss `>` VaR" and "loss `≥` VaR" — is unavoidable in exactly the setting where these tools must run.

## Baselines

- **Markowitz mean-variance (1952).** The template for "risk measure inside an optimization." Risk is the return variance `σ²(w) = wᵀ Σ w`; one minimizes `wᵀ Σ w` subject to a return floor `μᵀ w ≥ R*` and budget `Σ wᵢ = 1`, a convex quadratic program with a clean efficient frontier. *Gap:* variance penalizes upside and downside symmetrically and is a faithful tail measure only when returns are elliptical/normal; under fat tails or scenario discreteness it under-weights rare catastrophes, which are precisely the losses tail-risk management cares about. It controls dispersion, not the tail.

- **Parametric VaR (RiskMetrics, 1994–96).** Assume returns are jointly normal; then `VaR_α ∝ σ`, recovering mean-variance in disguise — convex and tractable, but only because it collapses to a standard deviation. *Gap:* the normality assumption fails for options and fat-tailed assets; it is just variance relabeled, with the same tail-blindness.

- **Historical / Monte-Carlo simulation VaR.** Resample or simulate `N` scenarios, compute the loss in each, read off the `⌈αN⌉`-th order statistic. Handles nonlinear instruments and arbitrary distributions. *Gap:* as an *objective in `w`* it is the non-convex, multi-extremal, tail-blind order statistic of the construction above — fine for *reporting* a fixed portfolio's risk, unusable for *choosing* the portfolio.

- **Axiomatic coherent measures (Artzner et al. 1999).** Establish the four axioms and that VaR violates subadditivity; propose tail/worst conditional expectation as coherent. *Gap:* characterized abstractly via generalized scenarios, with no tractable optimization-in-`w` formulation — a target to hit, not yet a method.

- **Mean shortfall / mean excess loss.** `E[ L | L > VaR_α ]`. The right concept (tail average), but defined through VaR, hence circular for optimization, and not coherent under discreteness when the threshold carries an atom.

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
        # TODO: any auxiliary decision variables the tail measure needs

    def _feasible_region(self, market_neutral=False):
        cons = [self.w >= self.lower, self.w <= self.upper]
        cons.append(cp.sum(self.w) == (0 if market_neutral else 1))
        return cons

    def _tail_risk_expr(self):
        # TODO: a CONVEX expression in self.w (and auxiliaries) for the
        # level-alpha tail-risk of the loss, computed from the scenario panel.
        # This is the slot the method fills.
        raise NotImplementedError

    def min_tail_risk(self, market_neutral=False):
        objective = self._tail_risk_expr()        # TODO
        constraints = self._feasible_region(market_neutral)
        # TODO: constraints that define the tail-risk auxiliaries
        cp.Problem(cp.Minimize(objective), constraints).solve()
        return self.w.value

    def efficient_return(self, target_return, market_neutral=False):
        # minimise tail risk s.t. expected return >= target. TODO: same slot.
        raise NotImplementedError
```

The method to come fills `_tail_risk_expr` and supplies the constraints that pin down its auxiliary variables; everything else (the panel, the weight variable, the budget/box region, the solver call) is already in place.
