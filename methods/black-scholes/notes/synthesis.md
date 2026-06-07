# Synthesis — Black–Scholes–Merton option pricing

## Sources actually read this run (in refs/ and src/)
- `black-1989-how-we-came-up.ocr.txt` — Fischer Black, "How We Came Up With the Option Formula," J. Portfolio Mgmt 1989. DIRECT first-person self-account. Backbone for the discovery order.
- `scholes-nobel-lecture-1997.ocr.txt` — Myron Scholes, "Derivatives in a Dynamic Environment," Nobel lecture 1997. Gives the exact CAPM-derivation equations (Taylor expansion, two-strategy table, the PDE, the discount-rate insight).
- `merton-nobel-lecture-1997.ocr.txt` — Robert Merton, "Applications of Option-Pricing Theory: 25 Years Later," Nobel lecture 1997. His continuous-trading replication/no-arbitrage derivation (Itô lemma, replicating portfolio, drift α cancels; investors need not agree on α).
- `black-scholes-1973-jpe.pdf` (read directly, pp.637-644) — PRIMARY: the exact PDE eq(7), hedge ratio 1/w₁, equity change eq(5), heat-equation substitution eq(9-12), final formula eq(13), Sprenkle/Samuelson/Thorp-Kassouf critiques.
- `merton-1973-rational-option-pricing.ocr.txt` — Merton, "Theory of Rational Option Pricing," Bell J. 1973 (PRIMARY, antecedent of continuous-time arg).

## The exact final artifacts (grounded, must land on these)
BS PDE (BS1973 eq 7), with w(x,t) option value, x stock price, t time, r riskless rate, v²=σ² variance rate:
  w₂ = r·w − r·x·w₁ − ½·v²·x²·w₁₁
boundary w(x,t*) = max(x−c, 0).
Substitution (eq 9) maps to heat equation y₂ = y₁₁ (eq 10), solved (Churchill 1963 p.155) → eq 13:
  w(x,t) = x·N(d₁) − c·e^{r(t−t*)}·N(d₂)
  d₁ = [ln(x/c) + (r + ½v²)(t*−t)] / (v·√(t*−t))
  d₂ = d₁ − v·√(t*−t) = [ln(x/c) + (r − ½v²)(t*−t)] / (v·√(t*−t))
N = standard normal CDF. Hedge ratio = w₁ = N(d₁) (shares per option).

## The central obstacle (from Black 1989 + Scholes 1997)
Pre-method warrant pricing (Sprenkle 1961, Boness 1964, Samuelson 1965) discounted the expected terminal value. TWO unknowns:
1. you need the stock's EXPECTED RETURN to compute the warrant's expected terminal value;
2. you need a DISCOUNT RATE for the warrant — but no single rate works, because the warrant's risk (hence its discount rate) varies with stock price and time.
Sprenkle had unknown k (expected appreciation ratio) and k* (discount factor), "tries to estimate them empirically but finds he is unable to do so." Samuelson had unknown α (stock return) and β (warrant discount rate), "no model of pricing under capital market equilibrium that would make this appropriate."

## The discovery path (Black 1989 order — the backbone)
1. CAPM background via Jack Treynor (1961a,b) at Arthur D. Little; Treynor's "value equation" (a differential equation for cash-flow valuation) had an ERROR — omitted second-derivative terms; Black learned to put them back. (Black 1989.)
2. Black writes warrant value as w(x,t), applies CAPM at every instant: warrant's expected return relates to its risk (beta) the same way a stock's does. CAPM → the discount rate for the warrant as function of x,t.
3. This gives a differential equation (notes dated June 1969). "It has just one solution" given terminal value + a condition he "didn't know about at the time."
4. Black fascinated: the warrant value does NOT depend on the stock's expected return, NOR on how risk splits into diversifiable/non-diversifiable — only on TOTAL risk (σ). "That fascinated me."
5. Black STUCK: spends "many, many days" trying to solve it. Has applied-math PhD + physics AB but "didn't recognize the equation as a version of the heat equation." Puts it aside.
6. Scholes joins (1969). Key move: since formula depends on σ not expected return, "we could solve the problem using any expected return for the stock." Choose expected return = riskless rate r (stock beta = 0, all risk diversifiable).
7. With r as the stock's drift and constant σ → lognormal terminal value. Use Sprenkle's formula for expected terminal option value, plugging r for the stock return.
8. Need PRESENT value, not expected terminal value → need a discount rate. "Rather suddenly, it came to us": if stock has beta 0 then option has beta 0, so option also earns r, so the DISCOUNT RATE is just r — constant, not stock-price/time dependent. Discount expected terminal value at r → the formula.
9. Check formula against the differential equation → it fits. "We knew it was right." Puts follow with small changes.
10. Merton's contribution (acknowledged in BS1973 footnote 3, "This was pointed out to us by Robert Merton"): with CONTINUOUS trading the stock+option hedge is LITERALLY riskless; the final published derivation uses this because it relies only on no-arbitrage, not CAPM.

## The two derivations (must reconstruct both — main text gives Merton's; CAPM is Scholes's lecture)
### A. CAPM / zero-beta derivation (Scholes Nobel lecture — the path they actually walked first)
Taylor expand w(x,t) over short Δt, keep second order in x (Itô):
  Δw = w₁Δx + w₂Δt + ½w₁₁Δx²Δt   [Scholes writes wₓΔx + w_t Δt + ½w_xx Δx²; Δx²→x²σ²Δt]
Two strategies with equal investment: (1) buy warrant w + bonds (w₁x−w) earning r; (2) buy w₁ shares of stock (w₁x). Both have the SAME risk (only uncertain term is Δx, same in both). No-arbitrage ⇒ equal returns. Equate, substitute Δx²→x²σ²Δt:
  r·w + w₁·x·r·... → rearranges to:  r·w + w₂ + r·x·w₁ + ½·w₁₁·x²·σ² = 0  (Scholes's sign convention; equals BS eq 7).
Expected stock return α absent. AMAZED.
### B. Riskless-hedge / no-arbitrage derivation (BS1973 main text — the published one, Merton's exactness)
Hold 1 share long, short 1/w₁ options. Equity = x − w/w₁ (eq 2). ΔEquity = Δx − Δw/w₁ (eq 3). By Itô (eq 4): Δw = w₁Δx + ½w₁₁v²x²Δt + w₂Δt. Substitute: ΔEquity = −(½w₁₁v²x² + w₂)Δt/w₁ (eq 5) — the Δx term CANCELS, so the change is deterministic → riskless. Riskless ⇒ must earn r·Δt on the equity: −(½w₁₁v²x²+w₂)Δt/w₁ = (x−w/w₁)·r·Δt (eq 6). Rearrange → PDE eq 7. Drift never entered.
### Solve (BS1973 eq 9-13)
Substitution eq 9 → heat equation y₂=y₁₁ (eq 10), boundary eq 11, solution by Churchill 1963 p.155 (eq 12) → formula eq 13. This is the heat equation Black failed to recognize.

## Design-decision → why table
- Why write w as a function w(x,t)? (vs discounting expected terminal value) — Treynor's approach; lets you apply Itô/Taylor and get a PDE; the discounting approach is stuck on the unknown discount rate that varies with x,t.
- Why keep the ½w₁₁Δx² term? — Δx is O(√Δt) so Δx²=O(Δt), same order as Δt; dropping it is wrong (this is exactly Treynor's omitted-second-derivative error). Itô.
- Why does Δx²→x²σ²Δt? — variance rate proportional to square of stock price (assumption b: lognormal / geometric BM); var of Δx over Δt is x²σ²Δt.
- Why geometric (lognormal) and not Bachelier's arithmetic BM? — arithmetic BM lets price go negative and gives constant absolute variance; stocks have ~constant percentage variance and limited liability (Samuelson 1965 fix of Bachelier 1900).
- Why short exactly 1/w₁ options per share (or w₁ shares per option)? — to first order Δw = w₁Δx, so w₁ shares move dollar-for-dollar with one option; this neutralizes the Δx risk.
- Why is the hedge riskless only with CONTINUOUS rebalancing? — w₁ changes as x,t move; over a finite interval the second-order curvature (w₁₁) leaves residual risk; continuous trading drives it to zero (Merton's point). Without continuous trading the residual is diversifiable (zero market covariance).
- Why must the riskless hedge earn r? — no-arbitrage: else borrow to build hedges and earn riskless excess; arbitrage forces return = r.
- Why does the expected return α drop out? — the hedge cancels the Δx term, which is the only place α lived (α multiplies Δt inside Δx's drift, but Δx enters identically in both legs / cancels). So the PDE has only r and σ.
- Why can we set α=r to get the formula (CAPM path)? — since the answer is independent of α, pick the convenient value; α=r is the zero-beta/risk-neutral choice that makes the option also earn r so the discount rate is the constant r. (This is the seed of risk-neutral valuation.)
- Why is the discount rate constant = r in the zero-beta world? — option beta = stock beta = 0 ⇒ option expected return = r at every instant ⇒ discount at r.
- Why N(d₁), N(d₂) form? — solving the heat equation = integrating the lognormal payoff; the two normal CDFs come from the cutoff lognormal expectation (Sprenkle's form) with k=k*=appropriate values once r is used for both drift and discount.
- Why r+½v² in d₁ vs r−½v² in d₂? — the ½v² is the Itô/lognormal mean-correction; d₂ is the (risk-neutral) probability of exercise, d₁ carries the extra ½v² from the x·(lognormal) term.
- Boundary/terminal condition w(x,t*)=max(x−c,0) — definition of a European call payoff; the "other condition Black didn't know" needed for uniqueness.

## In-frame discipline
Narrator = Black-then-Black+Scholes, present tense, 1969-1972. Antecedents to cite as prior art: Bachelier 1900, Samuelson 1965, Samuelson-Merton 1969, Sprenkle 1961, Boness 1964, Treynor 1961, Sharpe 1964/Lintner 1965 (CAPM), Thorp-Kassouf 1967, Itô calculus (via McKean 1969), Churchill 1963. Do NOT name "Black-Scholes 1973" or "this paper." Merton's continuous-trading insight enters as the collaborator's contribution that sharpens the hedge to exactly riskless (acknowledged in BS1973 fn3).
