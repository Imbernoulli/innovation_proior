# Context: deploying a mean-variance portfolio on a real trading desk

## Research question

The continuous mean-variance rule is settled: given expected returns `mu` (length `N`) and a
covariance matrix `Sigma` (`N x N`, symmetric positive semidefinite), choose weights `x` solving

```
min  x' Sigma x  -  q * mu' x      s.t.   sum_i x_i = 1,   l_i <= x_i <= u_i,
```

a convex quadratic program whose solution traces the efficient frontier as the risk tolerance `q`
sweeps. The output is a vector of fractional weights. The question here is narrower and entirely
operational: **what does a portfolio manager actually deploy?** A desk cannot trade the bare
frontier weights. Three facts about the real world break the continuous answer:

- A fund must often hold **at most `K` names** out of the `N` in the universe — far fewer than the
  optimizer's typical spread of many tiny positions. This is forced by monitoring and oversight cost,
  by focused-fund prospectuses that cap the holding count, by the wish not to be seen as a closet
  indexer, and by small separate accounts where per-name fixed costs dominate.
- A position that is held at all should be **at least some minimum size** `l_i` (a "buy-in"
  threshold). A 0.02% weight is operationally pointless and still pays a ticket charge. So each
  weight is *semi-continuous*: either `x_i = 0`, or `x_i` lies in `[l_i, u_i]`.
- Trades happen in **whole shares** (or whole round lots of, say, 100 shares). The deployable object
  is an integer share count `n_i`, not a real-valued weight; the realised weight is
  `x_i = n_i * p_i / V` with price `p_i` and budget `V`.

And because portfolios are *rebalanced*, not built from cash each period, the choice must also charge
for **moving** from the current book `x0` to the new one — proportional and fixed transaction costs,
and possibly a cap on total turnover.

A solution must (1) honour a hard upper bound on the number of holdings, (2) enforce a floor on every
nonzero position, (3) deliver tradeable integer share counts, and (4) net out the cost of rebalancing
— all while keeping the mean-variance objective that made the continuous rule sensible in the first
place. The continuous QP can express none of (1)-(3): each is a *disjunction* ("zero or at least
`l_i`", "at most `K` of them nonzero"), and disjunctions are not convex.

## Background

The field state is the Markowitz mean-variance framework (Markowitz 1952, 1959): summarise a
portfolio by its expected return `mu' x` (linear) and its variance `x' Sigma x` (a convex quadratic,
since `Sigma` is a covariance matrix and hence positive semidefinite), and trade the two off. With
linear budget and box constraints the feasible set is a convex polytope and the problem is a convex
QP, solvable to global optimality by mature pivoting / interior-point machinery. Roy's (1952)
safety-first rule and the CAPM line (Sharpe 1964, Lintner 1965) sit on the same two-moment summary.
This is the prior art that *works* — for fractional weights.

Several established facts constrain any extension to the discrete world:

- **The mean-variance objective is convex; the new constraints are not.** "Hold at most `K`
  names" and "each held name is at least `l_i`" each carve the simplex into a *union* of lower-
  dimensional faces — one face per subset of names allowed to be nonzero. A union of convex pieces is
  not convex. So the deployable feasible set is combinatorial, and the convex QP solver, which
  assumes one convex region, cannot represent it.

- **Mixed-integer linear programming already exists and is mature.** By the 1990s, branch-and-bound
  and branch-and-cut for integer and mixed-integer *linear* programs are standard: relax the
  integrality to get a convex (linear) bound, branch on a fractional integer variable into two
  subproblems, prune by bound, and add valid inequalities (cuts) to tighten the relaxation. The
  open question at the time is the *quadratic*-objective analogue — mixed-integer programs whose
  continuous relaxation is a convex QP rather than an LP have received comparatively little attention.

- **A continuous variable can be switched on and off by a binary indicator.** The fixed-charge /
  semi-continuous modelling trick is standard in operations research: to say "`x_i = 0` or
  `l_i <= x_i <= u_i`", attach a binary `y_i` and the pair of linear inequalities
  `l_i * y_i <= x_i <= u_i * y_i`. When `y_i = 0` both bounds collapse to `x_i = 0`; when `y_i = 1`
  they reopen to `[l_i, u_i]`. The upper bound `u_i` plays the role of the "big `M`" that, when the
  indicator is off, clamps the variable to zero. This is the bridge from a disjunction to linear
  constraints plus an integer variable.

- **A separable transaction cost is convex in the trade only if it has no fixed charge.** Splitting
  a trade `x_i - x0_i` into a buy part and a sell part, `x_i - x0_i = b_i - s_i` with `b_i, s_i >= 0`,
  turns the proportional cost `tc^+_i b_i + tc^-_i s_i` and the turnover `b_i + s_i = |x_i - x0_i|`
  into linear expressions (at the optimum only one of `b_i, s_i` is nonzero, so the split is exact).
  But a *fixed* per-trade fee — a charge `beta_i` the moment `x_i != x0_i`, regardless of size — is a
  step function of the trade, hence non-convex, and again needs a binary indicator to model.

- **The minimum-trading-lot / round-lot constraint is genuinely integer.** Shares trade in whole
  units; lots in fixed blocks. Replacing the real weight by an integer count `n_i` (with
  `x_i = n_i * s_i * p_i / V` for lot size `s_i`, price `p_i`, budget `V`) makes the variable itself
  integer rather than merely indicator-switched. Naively rounding the continuous weights breaks the
  budget `sum_i x_i = 1` and can violate the cardinality bound, so the rounding has to be done by an
  optimization that respects the budget and minimises the tracking error to the target weights.

The pain point is the gap between the convex frontier and what clears: the convex QP returns a dense,
fractional, costless-to-reach portfolio, while the deployable object is a *sparse, floored,
integer-share, rebalancing-cost-aware* one.

## Baselines

- **Continuous mean-variance QP (Markowitz 1952, 1959).** `min x' Sigma x - q mu' x` over the budget
  simplex with box bounds. Convex, globally solvable, traces the frontier. **Gap:** routinely returns
  many small nonzero weights, no mechanism to cap the holding count or to floor a position, and no
  notion of integer shares — so its optimum is, in general, not deployable as-is.

- **Solve the continuous QP, then heuristically prune.** Solve the convex problem, keep the `K`
  largest weights, set the rest to zero and renormalise (and/or round to whole shares). Cheap and
  common. **Gap:** the truncated portfolio is not optimal for the *constrained* problem — the best
  `K`-name portfolio need not be the top `K` names of the unconstrained one, because dropping a name
  changes the optimal weights of the survivors through the covariances; renormalisation can also
  violate the floor or the box, and rounding can break the budget.

- **Linearise the objective, then use integer linear programming.** Approximate `x' Sigma x` by a
  linear or piecewise-linear function (e.g. mean-absolute-deviation in place of variance), or assume
  equal weights across the chosen names, to land on a pure MILP that mature integer-LP solvers handle
  (Konno-Yamazaki 1991; and the equal-weight 0-1 reductions). **Gap:** it discards the genuine
  quadratic risk — the covariance structure that made diversification meaningful — to fit the linear
  tooling; the answer optimises a proxy, not the variance.

- **Tailored branch-and-bound on the cardinality-bounded QP (Bienstock 1996).** Keep the quadratic
  objective and impose the holding limit directly. Rather than introduce binaries, replace
  `|{i : x_i > 0}| <= K` with the surrogate inequality `sum_i (x_i / u_i) <= K` and branch on the
  continuous variables themselves — add `x_j <= 0` when branching down, `x_j >= alpha_j` when
  branching up — solving each node's convex QP with a primal-feasible descent method warm-started by a
  quadratic penalty, inside a branch-and-cut framework. **Gap / cost:** it is a specialised solver,
  tied to a particular node solver and surrogate; it solves the cardinality limit and minimum-position
  floor but is not framed around the full rebalancing problem (turnover, fixed ticket costs, round
  lots) in one statement.

- **Metaheuristics on the constrained frontier (Chang–Maringer–Beasley–Sharaiha lineage; genetic
  algorithms, tabu search, simulated annealing, local search).** Encode the chosen subset and search
  over subsets, solving the continuous QP within a fixed subset. **Gap:** no optimality guarantee —
  they return an *approximate* constrained frontier, with quality depending on the search budget.

## Evaluation settings

The natural inputs and yardsticks that exist before any discrete rule:

- **Inputs.** Expected returns `mu` (length `N`); covariance `Sigma` (`N x N`, symmetric PSD); a
  holding-count cap `K`; per-name floors `l_i` and caps `u_i`; the current holdings `x0` (or weights
  `w_prev`); proportional cost rates and any fixed per-trade fees; latest prices `p`, lot sizes `s`,
  and a total budget `V` for the round-lot step.
- **Feasible set.** The budget simplex `sum_i x_i = 1` intersected with the *semi-continuous* and
  *cardinality* structure: each `x_i` is `0` or in `[l_i, u_i]`, at most `K` of them nonzero; for the
  deployable allocation, share counts `n_i` are non-negative integers and the spend `sum_i n_i p_i`
  cannot exceed `V`.
- **Reportable quantities.** Expected return `mu' x`; variance `x' Sigma x`; the holding count
  `sum_i 1[x_i > 0]`; turnover `sum_i |x_i - x0_i|`; total transaction cost; leftover cash `V -
  sum_i n_i p_i`; and the tracking error between the deployed integer portfolio and the target weights.
- **Illustrative scale.** Small universes (a handful to a few tens of names) are where the discrete
  structure can be inspected directly — whether the floor binds, whether exactly `K` names survive,
  whether the lots track the target — and the method must then carry to the hundreds-of-names scale
  where branch-and-bound does the work.

## Code framework

The pre-existing machinery is the continuous mean-variance QP plus a convex/integer solver. The
weights, moments, budget, and box bounds exist; what does not yet exist is the rule that confines the
portfolio to a bounded number of floored, whole-lot positions and charges for rebalancing.

```python
import numpy as np
import cvxpy as cp

# ---- pre-existing inputs --------------------------------------------------
# mu      : np.ndarray (N,)      expected returns
# Sigma   : np.ndarray (N, N)    covariance matrix (symmetric PSD)
# bounds  : (l, u) per-name lower/upper weight bounds
# x0      : np.ndarray (N,)      current holdings (for rebalancing)
# prices  : np.ndarray (N,)      latest prices;  V : total budget;  lot size s

def portfolio_return(x, mu):
    return mu @ x                       # mu' x   (linear, known)

def portfolio_variance(x, Sigma):
    return cp.quad_form(x, Sigma, assume_PSD=True)   # x' Sigma x  (convex)


class DiscretePortfolioChooser:
    """Maps (mu, Sigma) + budget/box to a DEPLOYABLE portfolio:
    a bounded number of floored, whole-lot positions, net of trading cost.
    The discrete selection RULE is the empty slot below."""

    def __init__(self, mu, Sigma, weight_bounds=(0, 1)):
        self.mu, self.Sigma = mu, Sigma
        self.n = len(mu)
        self.x = cp.Variable(self.n)    # post-trade weights
        self.constraints = [cp.sum(self.x) == 1,
                            self.x >= weight_bounds[0],
                            self.x <= weight_bounds[1]]

    def choose(self, *args, **kwargs):
        # TODO: confine the portfolio to AT MOST K names, each either 0 or
        #       above a floor, and net out the cost of rebalancing from x0.
        #       What variables and constraints express "zero or at least l_i"
        #       and "no more than K nonzero", on top of the quadratic objective?
        raise NotImplementedError

    def deploy(self, weights, prices, total_value, lot_size=1):
        # TODO: turn target weights into tradeable WHOLE-LOT share counts
        #       under the budget, as close to the target as possible.
        raise NotImplementedError


def solve(objective, constraints, solver=None):
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)
    return prob
```
