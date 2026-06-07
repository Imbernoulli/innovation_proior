## Research question

An investor with a fixed amount of money and a list of securities must decide how to split the money
among them. The future returns of those securities are not known; they are uncertain. The question
is: what *rule* should the investor follow to choose the split — and can that rule be stated
precisely enough to compute the answer for a realistic number of securities, given some beliefs
about how the securities will perform?

The rule has to clear a bar that any candidate must meet: it must be consistent with the one
investment behavior everyone already agrees is sensible, namely diversification — spreading money
across several securities rather than betting it all on one. A rule that, taken at face value, would
tell a rational investor to concentrate everything in a single security cannot be the right rule,
because that is not what sensible investors do and not what a careful investor *should* do. So the
problem is twofold: find a decision rule under uncertainty that (a) explains and recommends
diversification, and (b) is computable in practice for many securities.

## Background

The tool on the table for valuing a security is the discounted-cash-flow idea, given its sharpest
form by John Burr Williams in *The Theory of Investment Value* (Harvard, 1938; his 1937 thesis).
Williams argued that the intrinsic value of a stock equals the present value of its future dividend
stream — discount each future dividend back to today and sum. Since dividends are paid over time and
into the future, this turns valuation into an exercise in discounting an expected flow. Read as a
decision rule, it suggests: value each security as the (discounted) return you anticipate from it,
and prefer the securities of greatest value. J. R. Hicks (*Value and Capital*, 1939) noted, for a
firm, that one could fold a risk allowance into the "anticipated" return or vary the discount rate
with risk — but the object is still a single number per security.

Sitting underneath any decision rule for uncertainty is a simple operational assumption about
beliefs: the investor acts as if carefully considered uncertain returns have probabilities attached
to them, even when those probabilities are partly judgmental rather than objective frequencies. Von
Neumann and Morgenstern (*Theory of Games and Economic Behavior*, 1944) had also shown how a
consistent agent under risk can be represented as maximizing expected utility, E[U]. That gives a
broad rational-choice ideal, but it does not yet give a portfolio rule that can be estimated and
computed from a manageable set of beliefs.

Two empirical facts frame the problem. First, diversification is universal and proverbial — "don't
put all your eggs in one basket" — yet it has no mathematical content: the proverb does not say how
much to spread, across which securities, or what measurable quantity spreading improves. Second, the
obvious mathematical gloss on the proverb — that the law of large numbers makes a diversified
portfolio's realized yield converge to its expected yield, so spreading "averages out" the risk — is
false for securities, because security returns are heavily intercorrelated; they do not behave like
independent draws, and no amount of spreading can drive the dispersion of portfolio return to zero.

The elementary statistics needed are standard. For a random variable Y with values yₖ and
probabilities pₖ, the expected value is E(Y) = Σ pₖ yₖ and the variance is V(Y) = Σ pₖ (yₖ − E Y)²,
with standard deviation σ = √V. For a weighted sum R = Σ aᵢ Rᵢ of random variables, the expected
value is the weighted sum of expected values, E(R) = Σ aᵢ E(Rᵢ); but the variance is *not* the
weighted sum of variances — it involves the covariances σᵢⱼ = E[(Rᵢ − E Rᵢ)(Rⱼ − E Rⱼ)] = ρᵢⱼ σᵢ σⱼ,
where ρᵢⱼ is the correlation coefficient.

## Baselines

**Maximize expected (discounted) return.** Taking Williams' valuation as a decision rule under
uncertainty: assign each security i its anticipated discounted return Rᵢ, and choose weights
Xᵢ ≥ 0 with Σ Xᵢ = 1 to maximize the portfolio's expected return R = Σ Xᵢ Rᵢ. Core idea: value is
expected discounted dividends; more value is better. The actual math: R is a weighted average of the
Rᵢ with nonnegative weights summing to one, so it is maximized by setting Xᵢ = 1 for the security
with the largest Rᵢ (ties broken arbitrarily). The specific gap: this rule never makes a diversified
portfolio preferable to all undiversified ones — it always sends the investor to a single security.
It contradicts the diversification everyone agrees on, so it fails as both a description of and a
guide to investment behavior. It also carries no notion of risk at all.

**Maximize expected return, then diversify by the law of large numbers.** A patch on the first rule:
among the securities tied for (or near) maximum expected return, spread the money, trusting that the
law of large numbers makes the portfolio's realized yield close to its expected yield. Core idea:
get the high mean *and* the safety of averaging. The gap: the law of large numbers requires
near-independence, and security returns are too intercorrelated for it to apply; diversification
cannot eliminate the variance. Moreover this rule presumes the maximum-expected-return portfolio is
also the minimum-dispersion portfolio, which is not generally true — there is a genuine trade-off
between expected return and dispersion, not a free lunch.

**Maximize expected utility E[U] over the full return distribution.** The broader rational-choice
rule from von Neumann–Morgenstern. Core idea: pick the portfolio whose return distribution has the
highest E[U] for the investor's utility U. The gap is operational, not conceptual: evaluating E[U]
for a portfolio of many securities requires the full joint distribution of returns, which is far
beyond what can be estimated and optimized with the data and computation available; it does not, by
itself, yield a procedure an investor can run.

## Evaluation settings

The natural inputs are beliefs about securities: per-security expected returns μᵢ, variances σᵢᵢ,
and pairwise covariances σᵢⱼ. These can be estimated from historical return series (e.g. the
per-period returns rᵢₜ for security i in period t) or formed by combining statistical estimates with
expert judgment. A candidate rule is exercised on small universes first — three securities in the
plane and four securities in three-space — and then in principle on larger universes of stocks and
funds, with constraints such as no short sales (Xᵢ ≥ 0) and full investment (Σ Xᵢ = 1). The relevant
outputs are the chosen split and the (expected-return, dispersion) pair it produces; the relevant
comparison is whether a rule recommends the diversified portfolios that sensible investors actually
hold. Historical annual and monthly return series for stocks and for funds are the data that such a
rule would be fed.

## Code framework

The available pieces are a way to hold beliefs (a mean vector and a covariance matrix), the algebra
for a portfolio's expected return and variance from weights, and a generic selection slot that turns
beliefs into candidate splits.

```python
import numpy as np

def portfolio_mean(weights, mu):
    # expected return of a weighted sum is the weighted sum of expected returns
    return weights @ mu

def portfolio_variance(weights, Sigma):
    # variance of a weighted sum carries all cross terms via the covariance matrix
    return weights @ Sigma @ weights

def select_portfolios(mu, Sigma, targets):
    # TODO: choose portfolio weights from beliefs and requested return levels.
    pass
```
