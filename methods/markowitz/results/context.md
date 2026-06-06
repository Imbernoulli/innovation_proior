# Context: turning beliefs about securities into a portfolio (circa 1950)

## Research question

An investor faces a universe of `N` securities. Suppose the first stage of the problem is
already done: from observation and experience the investor has arrived at *beliefs* about each
security's future return — for security `i`, an expected return `mu_i`, a variance `sigma_ii`,
and, for each pair, a covariance `sigma_ij`. The remaining problem is the second stage: turn
those beliefs into an actual choice of portfolio, i.e. a vector of weights `X = (X_1, ..., X_N)`
where `X_i` is the fraction of wealth placed in security `i`. Short sales are excluded, so
`X_i >= 0`, and the budget is fully invested, so `sum_i X_i = 1`.

The precise goal is a *rule* that maps `(mu, sigma)` to a portfolio — one that (1) is well
defined and computable, (2) is consistent with the plainly observed and sensible practice of
*diversification* (real investors spread their money; a rule that never recommends spreading
must be wrong), (3) captures the obvious fact that an investor likes high return and dislikes
risk, and (4) does so without first having to write down and estimate a full subjective utility
function over wealth, which no one knows how to elicit. A rule meeting all four is what is
missing. The candidate rules on the table each fail at least one.

## Background

The dominant prescription for valuing and choosing securities is the *present-value* / dividend-
discount view (J. B. Williams, *The Theory of Investment Value*, 1938; the discounting tradition
going back through Hicks 1939). One forms, for each security, the discounted value of its
anticipated stream of returns, `R_i = sum_t d_{it} r_{it}`, and prefers securities with higher
`R_i`. Risk enters only as a premium added to the discount rate. Williams's argument treats broad
diversification as enough to offset losses across securities, leaving effectively no residual risk.
A second strand reaches the same place through the *law of large
numbers* (a temptation since Jacob Bernoulli, 1713): spread money across many securities and the
realized average yield will converge to the expected yield, so risk washes out in the aggregate.

Several established facts about probability and about the world constrain any rule:

- **A portfolio return is a weighted sum of random variables.** With weights `X_i` fixed by the
  investor and returns `R_i` random, `R = sum_i X_i R_i`. The expected value of a weighted sum is
  the weighted sum of expected values, `E = sum_i X_i mu_i` — linear and simple. The *variance* of
  a weighted sum is not simple: it involves the covariances between every pair of constituents.
- **Covariance is real and large among securities.** Security returns are heavily intercorrelated
  — firms in the same industry tend to do badly at the same time. Empirically the returns are "too
  intercorrelated" for the law of large numbers to apply across a portfolio: diversification
  reduces dispersion but cannot drive it to zero, because the cross-correlations do not vanish.
  A portfolio rule therefore has to handle co-movement rather than assume it away.
- **Diversification is observed and is sensible**, and it is understood, informally, that its
  adequacy does not depend only on the *number* of holdings: sixty railway stocks are not as well
  diversified as a spread across railroads, utilities, mining, and manufacturing, because firms in
  the same industry have higher covariances among themselves.
- **Variance / standard deviation as a measure of dispersion** is standard mathematical statistics
  (Uspensky 1937 and any standard text). `V = sum_k p_k (y_k - E)^2`; `sigma = sqrt(V)`. Irving
  Fisher (1906) had already floated variance as a measure of economic risk, and Marschak (1938)
  had suggested means and the covariance matrix as a first-order description of preferences over
  uncertain consumption.

The pain point is the gap between the two strands above and reality: the present-value rule, taken
literally as a *guide*, gives degenerate (undiversified) advice; the law-of-large-numbers patch
rests on an empirically false premise about correlation. Neither yields the right kind of
diversification for the right reason.

## Baselines

- **Maximize discounted expected return (Williams 1938; Hicks 1939).** For each security compute a
  scalar `R_i` (discounted anticipated return); choose `X` to maximize `R = sum_i X_i R_i`. Because
  each `R_i` is fixed and independent of the weights, `R` is a weighted average of the `R_i` with
  non-negative weights summing to one; it is maximized by setting `X_a = 1` for whichever security
  has the largest `R_a` (or any mixture of ties). **Gap:** this rule *never* prefers a diversified
  portfolio to all undiversified ones — it always points at a single corner. Since diversification
  is both observed and sensible, the rule fails as a guide to behavior, no matter how the discount
  rates are chosen or how they vary over time.

- **Diversify among the maximum-expected-return securities (law of large numbers).** A patch:
  spread funds across all securities that share the maximum `mu_i`; invoke the law of large numbers
  so that the realized yield is almost the expected yield. **Gap:** this assumes a portfolio exists
  that simultaneously maximizes expected return and minimizes variance, and it assumes the law of
  large numbers applies across securities. Both are false in general — returns are too
  intercorrelated, and the maximum-`E` portfolio is generally *not* the minimum-`V` one.

- **Expected-utility maximization (the von Neumann–Morgenstern / Savage program, contemporaneous).**
  Posit a utility function `U(W)` over terminal wealth and maximize `E[U(W)]`. Coherent and general.
  **Gap:** it requires eliciting an investor's entire subjective utility function, which is not
  practical; it gives no directly computable portfolio rule from the inputs an analyst actually has
  (`mu`, `sigma`). One wants something operational that uses only first and second moments.

- **Safety-first (Roy, 1952, developed independently and in parallel).** Refuse to posit a utility
  function. Fix a "disaster level" return `d` the investor most wants to avoid falling below, and
  choose the portfolio that minimizes the probability `P(R < d)`. Under a two-moment description of
  the portfolio return, minimizing that probability amounts to maximizing the ratio `(mu_P - d)/sigma_P`
  — pushing the disaster level as many standard deviations below the mean as possible. Roy writes
  down the very same portfolio-variance-in-terms-of-covariances relation and arrives at a mean–
  variance efficient set. **Gap relative to the goal:** it commits to one specific point (the
  disaster-`d` ratio maximizer) and one specific risk attitude, rather than first laying out the
  whole menu of undominated risk–return trade-offs and letting the investor choose.

## Evaluation settings

The natural inputs and yardsticks that exist before any new rule:

- **Inputs.** A vector of expected returns `mu` (length `N`) and a symmetric positive-semidefinite
  covariance matrix `Sigma` (`N x N`), `Sigma_ij = rho_ij sigma_i sigma_j`. In practice these come
  from combining statistical estimates over some past period with the judgment of analysts; the
  question of how best to *form* the beliefs is the (excluded) first stage.
- **Feasible set.** The probability simplex: `X_i >= 0`, `sum_i X_i = 1` (the long-only, fully-
  invested budget). Optionally arbitrary lower/upper bounds `l_i <= X_i <= u_i`, and, if shorting
  is permitted, negative weights.
- **Reportable quantities of a portfolio.** Expected return `E = mu^T X`; variance `V = X^T Sigma X`;
  standard deviation `sigma_P = sqrt(V)`; and, given a reference rate `r`, the reward-to-variability
  ratio `(mu^T X - r)/sigma_P`. The geometric object of interest is the attainable set of `(E, V)`
  pairs over all feasible `X`, and within it the *efficient* boundary — minimum `V` for each level
  of `E`, equivalently maximum `E` for each level of `V`.
- **Illustrative scale.** Small cases (three and four securities) are the natural place to reason
  geometrically; the method must then generalize to arbitrary `N`, ideally with a procedure that
  computes the entire efficient boundary, not one point at a time.

## Code framework

The pre-existing machinery is just numerical linear algebra and a convex-optimization solver. The
weights, the moments, and the budget/sign constraints all exist; what does not yet exist is the
*choice rule* — the objective(s) and the procedure that selects portfolios from `(mu, Sigma)`.

```python
import numpy as np
import cvxpy as cp

# ---- pre-existing inputs (the investor's beliefs) -------------------------
# mu     : np.ndarray, shape (N,)      expected return of each security
# Sigma  : np.ndarray, shape (N, N)    covariance matrix (symmetric PSD)
# bounds : (l, u) per-weight lower/upper bounds; (0, 1) is long-only

def portfolio_return(w, mu):
    return w @ mu                      # E = mu^T w   (linear, known)

def portfolio_variance(w, Sigma):
    return cp.quad_form(w, Sigma, assume_PSD=True)   # V = w^T Sigma w

class PortfolioChooser:
    """Maps beliefs (mu, Sigma) + budget/sign constraints to a weight vector.
    The selection RULE is the empty slot below."""

    def __init__(self, mu, Sigma, weight_bounds=(0, 1)):
        self.mu, self.Sigma = mu, Sigma
        self.n = len(mu)
        self.w = cp.Variable(self.n)
        # budget + box constraints exist; the objective does not yet
        self.constraints = [cp.sum(self.w) == 1,
                             self.w >= weight_bounds[0],
                             self.w <= weight_bounds[1]]

    def choose(self, *args, **kwargs):
        # TODO: the rule that turns (mu, Sigma) into a portfolio.
        #       What scalar objective (or family of objectives) should w solve,
        #       and how do we sweep it to produce the whole menu of choices?
        raise NotImplementedError

    def frontier(self, n_points):
        # TODO: produce the set of undominated (return, risk) portfolios.
        raise NotImplementedError


def solve_qp(objective, constraints):
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    return prob
```
