## Research question

A call option gives its holder the right, but not the obligation, to buy one share of a
stock at a fixed exercise price `c` on a fixed maturity date `t*`. What is it worth today,
when the stock trades at `x` and there are `t*−t` years left? The payoff at maturity is
mechanical — `max(x−c, 0)`, worthless if the stock finishes below the strike, otherwise the
amount it finishes in-the-money. The difficulty is everything before maturity: the value
today depends on where the stock might go, and that is uncertain.

The natural recipe — take the expected payoff at maturity and discount it back to the
present — runs into two unknowns that the recipe itself cannot supply. To compute the
expected payoff you must know the stock's expected rate of return (how fast, on average,
the share drifts upward), and that number is notoriously unobservable and contested. To
discount the expected payoff you must know the rate at which to discount a claim as risky
as an option — and an option's riskiness is not a fixed number: it changes as the stock
moves and as maturity approaches, so no single discount rate is correct over the option's
life. A satisfactory theory would have to pin the value down from things one can actually
measure — the stock price, the exercise price, the time remaining, the interest rate, and
the stock's volatility — without smuggling in either of these two unknowns. The prize is
not just options: warrants, convertible bonds, and ultimately the equity and risky debt of
any levered firm are options in disguise, so a correct option formula would reprice a large
part of corporate finance.

## Background

**The price process.** Bachelier (1900), in his thesis on the theory of speculation, modeled
the stock price as Brownian motion — the increment over a short interval is normal with
variance proportional to the elapsed time. As a model of prices this has two defects: the
absolute variance is the same whether the stock is at \$5 or \$500, and the price can wander
negative. Samuelson (1965) repaired both by modeling the *logarithm* of the price as
Brownian motion (geometric Brownian motion): percentage returns, not dollar changes,
have constant variance, the price stays positive, and the terminal price is lognormally
distributed. Over a short interval `Δt` the proportional change `Δx/x` has mean `α·Δt` and
variance `σ²·Δt`; equivalently the variance of `Δx` is `x²σ²·Δt`. Here `α` is the stock's
expected return and `σ²` (written `v²` below) is the variance rate, taken constant.

**Itô's stochastic calculus.** When the underlying follows such a continuous random walk, a
smooth function `w(x,t)` of the price cannot be expanded by ordinary calculus. Because the
increment `Δx` is of order `√Δt`, its square `Δx²` is of order `Δt` — the same order as the
time step — so the second-order term cannot be dropped. The correct short-interval
expansion keeps it:
```
Δw = w₁·Δx + w₂·Δt + ½·w₁₁·Δx²,
```
where subscripts denote partial derivatives of `w` with respect to its first argument
(stock price) and second argument (time). Substituting the variance of `Δx`, the term
`½·w₁₁·Δx²` contributes `½·w₁₁·x²σ²·Δt` deterministically. This extra second-derivative term
is what ordinary calculus would drop. (An exposition of this calculus is McKean, 1969.)

**The Capital Asset Pricing Model.** Treynor (1961), and independently Sharpe (1964),
Lintner (1965) and Mossin, established a general-equilibrium relation: in equilibrium a
security's expected excess return is proportional to its *beta*, the covariance of its return
with the market divided by the market's variance. Only the non-diversifiable (market)
component of risk is rewarded; idiosyncratic risk, which a diversified investor can wash out,
earns nothing. A consequence of this relation: a security or portfolio with zero beta —
no covariance with the market — must, in equilibrium, earn exactly the riskless interest
rate, because none of its risk is the kind the market pays for. Treynor had also written a
"value equation," a differential equation for the present value of a company's uncertain
cash flows.

**No-arbitrage / the law of one price.** Modigliani and Miller had shown that two portfolios
with identical payoffs in every state must have identical prices, on pain of a riskless
money pump. If a combination of securities can be made to behave, instant by instant, like
a riskless bond, then it must earn the riskless rate — otherwise an investor could borrow at
`r`, build the combination, and pocket the difference with no risk. This principle is
weaker than full equilibrium (it needs no assumption about preferences or the market
portfolio) and therefore more robust.

**Observed properties of the option's price as a function of the stock.** Empirically and by
inspection, the value of a call rises with the stock price, lies below the 45° line `w ≤ x`,
is bounded below by `max(x−c, 0)` and above by `x`, is convex in the stock price, and decays
toward that lower bound as maturity nears. Deep in-the-money the option tracks the stock
nearly dollar-for-dollar; deep out-of-the-money it barely moves. Crucially, the option is a
*leveraged* claim: a given percentage move in the stock produces a larger percentage move in
the option, and the leverage — hence the option's risk and its beta — is not constant but
depends on the stock price and on time to maturity. This is precisely why a single discount
rate cannot value it.

## Baselines

**Sprenkle (1961).** Assumes the terminal stock price is lognormal and writes the option's
value as the expected payoff of the cut-off lognormal:
```
k·x·N(b₁) − k*·c·N(b₂),
b₁ = [ln(k·x/c) + ½v²(t*−t)] / (v·√(t*−t)),
b₂ = [ln(k·x/c) − ½v²(t*−t)] / (v·√(t*−t)).
```
Here `k` is the ratio of the stock's expected terminal price to its current price (it
encodes the unknown expected return) and `k*` is a discount factor for the option that
depends on the option's risk. Sprenkle tries to estimate `k` and `k*` empirically and
cannot. The functional form is already close to what a complete valuation would need, but
with two free knobs nailed to nothing.

**Boness (1964).** Improves Sprenkle by discounting the expected terminal value at the
stock's own expected rate of return. This at least supplies a discount rate, but the wrong
one: it presumes the option carries the same risk per dollar as the stock, which the
leverage argument shows is false, and it still requires the unknown expected return.

**Samuelson (1965), "rational warrant pricing."** Treats the warrant price as a function of
the stock price, assumes lognormal terminal stock value, takes the cut-off expected value,
and discounts it — but at a rate `β` for the warrant that differs from the stock's return
`α`, with both `α` and `β` left as free parameters. Samuelson states plainly there is no
model of pricing under capital-market equilibrium that would justify any particular choice.

**Samuelson and Merton (1969).** Recognize that discounting the cut-off expected value at an
exogenous rate is not an appropriate procedure, treat the option price as a function of the
stock price, and recognize that the discount rates are tied to investors' willingness to
hold the outstanding stock and option. But their resolution leans on the representative
investor's utility function: the final warrant price depends on the assumed shape of that
utility, so the answer is not preference-free.

**Thorp and Kassouf (1967).** Fit an empirical curve to observed warrant prices and use its
slope to form a hedged position — long the stock, short the warrant in the slope ratio — to
neutralize first-order moves. Their interest is empirical — identifying warrants that look
mispriced relative to the fitted curve — and they stop at the hedge itself; they extract no
closed-form valuation from it.

The common gap: every one of these either carries an unobservable expected-return parameter,
or an unjustified discount rate for a claim whose risk varies with price and time, or a
dependence on an assumed utility function. None prices the option from observables alone.

## Evaluation settings

A theory of this kind would be tested against market data on traded warrants and on
over-the-counter call options — for example, a time series of warrant prices and matched
underlying-stock prices, or a broker's transactions diary recording the premiums at which
over-the-counter options were actually written. The natural inputs are the stock price, the
exercise price, the time to maturity, the prevailing short-term interest rate, and an
estimate of the stock's volatility from historical returns. The natural diagnostics are
whether a hedged position formed from a candidate slope rule earns a return uncorrelated
with the market, and whether buying options a candidate valuation marks cheap and selling
those it marks dear is profitable before transaction costs. Listed-option markets of the kind that
would make continuous rehedging cheap do not yet exist; the over-the-counter market is thin
and its transaction costs are large.

## Code framework

A pricing worksheet can already compute the terminal payoff, describe the lognormal
terminal stock distribution under an assumed drift, and discount a future amount at a
chosen rate. The unresolved step is a valuation rule that no longer depends on an assumed
stock drift or on an option-specific discount rate.

```python
import numpy as np
from scipy.stats import norm

def call_payoff(x_T, c):
    return np.maximum(x_T - c, 0.0)

def lognormal_terminal_params(x, alpha, sigma, tau):
    mean_log = np.log(x) + (alpha - 0.5 * sigma**2) * tau
    std_log = sigma * np.sqrt(tau)
    return mean_log, std_log

def discount(value, rate, tau):
    return value * np.exp(-rate * tau)

def call_value(x, c, tau, r, sigma, alpha=None, discount_rate=None):
    # TODO: valuation rule using x, c, tau, r, and sigma.
    pass

def put_value(x, c, tau, r, sigma):
    # TODO: value the matching European put once the call rule is known.
    pass

def hedge_ratio(x, c, tau, r, sigma):
    # TODO: slope of the call value with respect to the stock price.
    pass
```
