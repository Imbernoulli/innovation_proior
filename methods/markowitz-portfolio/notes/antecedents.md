# Antecedents and lineage — research notes (grounded in retrieved sources)

## Self-account (backbone) — Markowitz, Nobel Lecture "Foundations of Portfolio Theory" (1990)
`refs/markowitz-nobel-lecture-1990.{pdf,txt}` (and .html). The decisive passage (txt lines 61–83):

> "The basic principles of portfolio theory came to me one day while I was reading John Burr
> Williams, *The Theory of Investment Value*. Williams proposed that the value of a stock should
> equal the present value of its future dividend stream. But clearly dividends are uncertain, so I
> took Williams' recommendation to be to value a stock as the **expected** value of its discounted
> future dividend stream. But if the investor is concerned only with the expected values of
> securities, the investor must also be only interested in the expected value of the portfolio. To
> maximize the expected value of a portfolio, one need only invest in one security — the security
> with maximum expected return ... Thus action based on expected return only ... must be rejected
> ... It seemed obvious that investors are concerned with risk and return ... Variance ... came to
> mind as a measure of risk of the portfolio. The fact that the variance of the portfolio, that is
> the variance of a weighted sum, involved all covariance terms added to the plausibility of the
> approach. Since there were two criteria — expected return and risk — the natural approach for an
> economics student was to imagine the investor selecting a point from the set of Pareto optimal
> expected return, variance of return combinations, now known as the efficient frontier."

Also from the lecture:
- "An investor who knew future returns with certainty would invest in only one security ... In no
  case would the investor actually prefer a diversified portfolio. But diversification is a common
  and reasonable investment practice. Why? To reduce uncertainty!" (txt lines ~46–52)
- Savage convinced him a rational agent under uncertainty acts on subjective/"probability beliefs"
  that combine like objective probabilities (Savage, *Foundations of Statistics*, 1954). (lines 54–60)
- The 1959 book relates mean-variance to von Neumann–Morgenstern and Savage expected utility; the
  mean-variance rule is defended as a computationally feasible *approximation* to expected-utility
  maximization (Levy–Markowitz correlations ~0.99 for diversified portfolios). (lines 84–91, 117–189)
- 1956: "critical line algorithm" for tracing the efficient frontier (Naval Research Logistics Q.).
- Friedman's quip at the thesis defense: "portfolio theory was not Economics." (lines 386–396)

## Primary source — Markowitz, "Portfolio Selection," J. Finance 7(1):77–91 (1952)
`refs/markowitz-1952-portfolio-selection.{pdf,txt}`. Structure / load-bearing content:
- Two-stage portfolio process; the paper is about stage two (beliefs → choice). (txt 45–58)
- Rule 1 (discounted/anticipated expected return maximization) — REJECTED because it never implies a
  diversified portfolio is preferable; R = Σ Xᵢ Rᵢ is a weighted average, maximized by putting Xᵢ=1
  on the max-Rᵢ security (short sales excluded, Σ Xᵢ = 1, Xᵢ ≥ 0). (txt 59–116)
- The "law of large numbers" patch rejected: returns are too intercorrelated; diversification cannot
  eliminate all variance; max-E portfolio ≠ min-V portfolio; there is a *rate* at which one trades E
  for V. (txt 129–144)
- Statistics primer: E(R)=Σ aᵢ E(Rᵢ); Var of a weighted sum needs covariance;
  σᵢⱼ = E[(Rᵢ−E Rᵢ)(Rⱼ−E Rⱼ)] = ρᵢⱼ σᵢ σⱼ. (txt 161–223)
- Portfolio: R = Σ Xᵢ Rᵢ, Xᵢ fixed by investor, Σ Xᵢ = 1, Xᵢ ≥ 0;
  E = Σ Xᵢ μᵢ, V = Σᵢ Σⱼ Xᵢ Xⱼ σᵢⱼ. (txt 224–247)
- E–V rule: choose from the *efficient* set — min V for given E (or more), max E for given V (or
  less); the set of attainable (E,V) is a region, its NW boundary is efficient. (txt 259–266)
- Geometry of the 3-security case: substitute X₃ = 1 − X₁ − X₂; isomean curves are parallel
  straight lines, isovariance curves are concentric ellipses centered at the global min-V point X̄;
  the locus of tangencies (critical line) is a straight line; the efficient set is a connected
  series of straight-line segments from min-V to max-E. (txt 295–407)
- In (E,V) space: E is a plane, V a paraboloid over the X-simplex; plotting V against E for efficient
  portfolios gives connected parabola segments. (txt 408–419)
- Why E–V gives the *right kind* of diversification: not many securities but LOW-covariance ones;
  diversify across industries (lower cross-industry covariance). (txt 466–490)
- Two-portfolio mixing P = λP' + (1−λ)P'': variance of the mix is typically below either, equal only
  if perfectly correlated; this is the convexity that makes diversification work. (txt 500–509)
- E–V as a working approximation; later (1959) tied to U(E,V), ∂U/∂E>0, ∂U/∂V<0. (txt 517–539)

## Antecedent 1 — John Burr Williams, *The Theory of Investment Value* (Harvard, 1938)
Williams (PhD thesis 1937): the intrinsic value of a stock = the present (discounted) value of its
future dividend stream ("algebraic budgeting"; the dividend-discount model, precursor to the Gordon
growth model). Cited in the 1952 paper (footnote 1, txt 77–78) and named as the *trigger* in the
Nobel lecture. KEY GAP it leaves: it is a *valuation* rule that, taken as a decision rule under
uncertainty (value = expected discounted dividends → maximize it), implies full concentration in the
single highest-value security. It has no notion of risk and therefore no rationale for diversifying.

## Antecedent 2 — the folk wisdom "don't put all your eggs in one basket"
Diversification was universally practiced and proverbial, but had no mathematical content: it didn't
say *how much* to diversify, *which* securities, or *why* spreading reduces something measurable. A
naive math gloss ("law of large numbers → average out the risk") is wrong because security returns
are correlated. The gap: practice without a quantitative criterion.

## Antecedent 3 — expected-utility theory (von Neumann–Morgenstern 1944; Savage 1954)
vNM axioms ⇒ a rational agent maximizes expected utility E[U(R)]; Savage extends this to subjective
probabilities. This is the "theoretically correct" object. GAP for an investor: maximizing E[U]
over the full distribution of an N-security portfolio is data-hungry (needs the whole joint
distribution / many moments) and computationally infeasible at the time. Markowitz's stance: prefer
a *feasible approximation* (two moments, E and V) to a precise-but-uncomputable one — and he later
shows (Levy–Markowitz) that f(E,V) tracks E[U] with correlation ~0.99 for diversified portfolios.

## Antecedent 4 — J. R. Hicks, *Value and Capital* (1939, p.126)
Cited (1952 footnote 2): one could fold a risk allowance into "anticipated" returns, applied to a
firm. GAP: still a single scalar per security; no portfolio-level risk, no covariance, no frontier.

## Two-asset frontier math (for the small worked example; grounded in retrieved sources)
- E_p = w μ₁ + (1−w) μ₂.
- σ_p² = w² σ₁² + (1−w)² σ₂² + 2 w(1−w) ρ σ₁ σ₂  (Wikipedia MPT; Sharpe two-asset notes).
- Global min-variance weight (two assets): w* = (σ₂² − ρσ₁σ₂) / (σ₁² + σ₂² − 2ρσ₁σ₂).
- Locus of (σ_p, E_p) as w varies is (one branch of) a hyperbola — the "Markowitz bullet"; the
  upper branch from the min-variance point is the efficient frontier.
- N-asset matrix form: E_p = wᵀμ, σ_p² = wᵀΣw, minimize wᵀΣw s.t. wᵀμ = target, wᵀ1 = 1.
  Closed form via Lagrange gives E ↦ V a parabola (V against E), i.e. (σ,E) a hyperbola — matches the
  1952 "connected parabola segments" once no-short-sale corners are smoothed away.
