# Synthesis — what reasoning.md must re-derive (Markowitz's own path)

## The pain point
A rational investor must choose a portfolio under uncertainty. The valuation tool on the table
(Williams 1938): a security's value = expected discounted future dividends. Read as a decision rule,
"maximize expected (discounted) return" is the only candidate. The discovery is the realization that
this rule is *wrong* in a way that the proverb everyone already follows exposes.

## The library moment (backbone, from the Nobel lecture)
Reading Williams. Take his rule literally under uncertainty: value = E[discounted dividends].
Portfolio value/return is then E[Σ Xᵢ Rᵢ] = Σ Xᵢ E[Rᵢ] = Σ Xᵢ μᵢ — a weighted average of the
per-security expected returns, with weights Xᵢ ≥ 0, Σ Xᵢ = 1. A weighted average is maximized at a
vertex: put everything on the single security with the largest μᵢ. So "maximize expected return"
**never** prefers a diversified portfolio. Yet diversification is universal and sensible. The rule
contradicts the most basic observed behavior. Why diversify? To reduce uncertainty.

## The wall and the patch that fails
First patch (the standard one): invoke the law of large numbers — diversify across the max-E
securities and the realized yield ≈ expected yield. Reject it: returns are too intercorrelated;
the LLN needs (near-)independence; diversification cannot drive variance to zero. And the max-E
portfolio is not the min-V portfolio — there is a *rate* at which you give up E to buy down V.

## The second axis
If uncertainty is what makes diversification rational, uncertainty must enter the objective.
A measure of dispersion of portfolio return: variance V (equivalently σ). Now there are two
criteria, E and V, and the killer fact: V of a weighted sum is NOT a weighted sum of variances —
it carries all the cross terms,
    V = Σᵢ Σⱼ Xᵢ Xⱼ σᵢⱼ,   σᵢⱼ = ρᵢⱼ σᵢ σⱼ.
The off-diagonal covariances are exactly what lets a mix have lower variance than its parts when
returns aren't perfectly correlated — this is why diversification reduces risk, and why it's
"right kind" of diversification (low-covariance, cross-industry), not just "many names." That the
covariances appear is what makes variance the *right* risk measure: it has diversification built in.

## Two criteria ⇒ a frontier, not a point
With two criteria and no a-priori trade-off rate, the economics-student move is Pareto efficiency:
keep only (E,V) pairs with min V for given E (or more) and max E for given V (or less). The
attainable (E,V) set is a region; its NW boundary is the efficient frontier. The investor picks one
point on this one-dimensional curve according to taste.

## The geometry (1952, 3-security case) — derive in-frame
Σ Xᵢ = 1 removes a dimension: X₃ = 1 − X₁ − X₂, so work in the (X₁,X₂) plane. The attainable set is
the triangle (Xᵢ ≥ 0). E is linear in (X₁,X₂) ⇒ isomean lines are parallel straight lines. V is a
positive-definite quadratic ⇒ isovariance curves are concentric ellipses centered at the global
min-V point X̄. Minimizing V on a fixed isomean line = the tangency of that line with an ellipse;
as E varies the tangency points sweep a straight line (the critical line). The efficient set runs
from X̄ (or the min-V boundary point) along critical lines to the max-E vertex — connected straight
segments. In (E,V) space: E plane, V paraboloid; V-against-E for efficient portfolios = connected
parabola segments (a single parabola/hyperbola once the no-short corners are gone).

## Land on
The mean-variance model {E = wᵀμ, V = wᵀΣw, minimize V s.t. wᵀμ = target, wᵀ1 = 1} and the efficient
frontier, with a small worked 3-security frontier (closed-form Lagrange sweep) showing (a) the
global min-variance portfolio is far safer than the max-mean asset, (b) a mix has lower σ than the
weighted-average standalone σ (covariance at work), (c) the (σ,E) curve V(E) = (A E² − 2BE + C)/D.

## Design choices → why (for reasoning.md to live out)
- Variance (not range, not "probability of loss"): it is the dispersion that (i) makes the portfolio
  risk a clean quadratic in weights, (ii) automatically pulls in covariances, so diversification is
  endogenous; standard deviation / coefficient of variation give the same efficient set (monotone).
  (Markowitz later flags semi-variance as the only serious alternative — adverse deviations only.)
- Two moments, not the full distribution / full expected utility: E[U] is the "correct" object but
  data-hungry and not computable at scale; (E,V) is feasible and, empirically, f(E,V) ≈ E[U] with
  correlation ~0.99 for diversified portfolios (Levy–Markowitz). Feasible-approximation over
  precise-but-uncomputable. [Keep as derivation-time *reason*, not a cited later result.]
- Efficiency/Pareto rather than one utility-maximizing point: avoids committing to the investor's
  unknown risk preference; hand them the whole frontier, they pick.
- No-short-sale (Xᵢ ≥ 0): realistic constraint that makes the attainable set the simplex and the
  efficient set piecewise-linear; the small example relaxes it to keep the parabola exact, noting
  the corners reappear with the constraint.

## In-frame discipline
First-person present tense; the narrator IS Markowitz in the library / at the desk. Never cite "the
1952 paper" / Nobel lecture / "the authors." Williams, Hicks, von Neumann–Morgenstern, Savage by
name as prior art is correct and stays. No hindsight (no CAPM, no Sharpe ratio, no "later work").
