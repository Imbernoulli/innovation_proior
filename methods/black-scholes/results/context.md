## Research question

A call option gives its holder the right, but not the obligation, to buy one share of a
stock at a fixed exercise price `c` on a fixed maturity date `t*`. What is it worth today,
when the stock trades at `x` and there are `t*−t` years left? The payoff at maturity is
mechanical — `max(x−c, 0)`, worthless if the stock finishes below the strike, otherwise the
amount it finishes in-the-money. The value before maturity depends on where the stock
might go, and that is uncertain.

The natural recipe is to take the expected payoff at maturity and discount it back to the
present. Carrying it out calls for the stock's expected rate of return (how fast, on
average, the share drifts upward) to compute the expected payoff, and a rate at which to
discount a claim as risky as an option. The available observables are the stock price, the
exercise price, the time remaining, the interest rate, and the stock's volatility. The
question is how to value a call — and the warrants, convertible bonds, and the equity and
risky debt of any levered firm that are options in disguise — from these quantities.

## Background

**The price process.** Bachelier (1900), in his thesis on the theory of speculation, modeled
the stock price as Brownian motion — the increment over a short interval is normal with
variance proportional to the elapsed time. Under this model the absolute variance is the
same whether the stock is at \$5 or \$500, and the price can wander negative. Samuelson
(1965) modeled the *logarithm* of the price as Brownian motion (geometric Brownian motion):
percentage returns, not dollar changes, have constant variance, the price stays positive,
and the terminal price is lognormally distributed. Over a short interval `Δt` the
proportional change `Δx/x` has mean `α·Δt` and variance `σ²·Δt`; equivalently the variance of
`Δx` is `x²σ²·Δt`. Here `α` is the stock's expected return and `σ²` (written `v²` below) is
the variance rate, taken constant.

**Itô's stochastic calculus.** When the underlying follows such a continuous random walk, a
smooth function `w(x,t)` of the price cannot be expanded by ordinary calculus. Because the
increment `Δx` is of order `√Δt`, its square `Δx²` is of order `Δt` — the same order as the
time step — so the second-order term is retained. The correct short-interval expansion is
```
Δw = w₁·Δx + w₂·Δt + ½·w₁₁·Δx²,
```
where subscripts denote partial derivatives of `w` with respect to its first argument
(stock price) and second argument (time). Substituting the variance of `Δx`, the term
`½·w₁₁·Δx²` contributes `½·w₁₁·x²σ²·Δt` deterministically. (An exposition of this calculus is
McKean, 1969.)

**The Capital Asset Pricing Model.** Treynor (1961), and independently Sharpe (1964),
Lintner (1965) and Mossin, established a general-equilibrium relation: in equilibrium a
security's expected excess return is proportional to its *beta*, the covariance of its return
with the market divided by the market's variance. Only the non-diversifiable (market)
component of risk is rewarded; idiosyncratic risk, which a diversified investor can wash out,
earns nothing. A consequence of this relation: a security or portfolio with zero beta —
no covariance with the market — earns exactly the riskless interest rate in equilibrium,
because none of its risk is the kind the market pays for. Treynor had also written a
"value equation," a differential equation for the present value of a company's uncertain
cash flows.

**No-arbitrage / the law of one price.** Modigliani and Miller had shown that two portfolios
with identical payoffs in every state must have identical prices, on pain of a riskless
money pump. If a combination of securities can be made to behave, instant by instant, like
a riskless bond, then it earns the riskless rate — otherwise an investor could borrow at
`r`, build the combination, and pocket the difference with no risk. This principle needs no
assumption about preferences or the market portfolio.

**Observed properties of the option's price as a function of the stock.** Empirically and by
inspection, the value of a call rises with the stock price, lies below the 45° line `w ≤ x`,
is bounded below by `max(x−c, 0)` and above by `x`, is convex in the stock price, and decays
toward that lower bound as maturity nears. Deep in-the-money the option tracks the stock
nearly dollar-for-dollar; deep out-of-the-money it barely moves. The option is a
*leveraged* claim: a given percentage move in the stock produces a larger percentage move in
the option, and the leverage — hence the option's risk and its beta — varies with the
stock price and the time to maturity.

## Baselines

**Sprenkle (1961).** Assumes the terminal stock price is lognormal and writes the option's
value as the expected payoff of the cut-off lognormal:
```
k·x·N(b₁) − k*·c·N(b₂),
b₁ = [ln(k·x/c) + ½v²(t*−t)] / (v·√(t*−t)),
b₂ = [ln(k·x/c) − ½v²(t*−t)] / (v·√(t*−t)).
```
Here `k` is the ratio of the stock's expected terminal price to its current price (it
encodes the expected return) and `k*` is a discount factor for the option that depends on
the option's risk. Sprenkle estimates `k` and `k*` empirically.

**Boness (1964).** Discounts the expected terminal value at the stock's own expected rate of
return, presuming the option carries the same risk per dollar as the stock.

**Samuelson (1965), "rational warrant pricing."** Treats the warrant price as a function of
the stock price, assumes lognormal terminal stock value, takes the cut-off expected value,
and discounts it at a rate `β` for the warrant that differs from the stock's return `α`,
with both `α` and `β` left as free parameters.

**Samuelson and Merton (1969).** Treat the option price as a function of the stock price and
tie the discount rates to investors' willingness to hold the outstanding stock and option,
deriving the warrant price from the representative investor's utility function, so that the
final price depends on the assumed shape of that utility.

**Thorp and Kassouf (1967).** Fit an empirical curve to observed warrant prices and use its
slope to form a hedged position — long the stock, short the warrant in the slope ratio — to
neutralize first-order moves. Their aim is empirical: identifying warrants that look
mispriced relative to the fitted curve.

## Evaluation settings

A theory of this kind is tested against market data on traded warrants and on
over-the-counter call options — for example, a time series of warrant prices and matched
underlying-stock prices, or a broker's transactions diary recording the premiums at which
over-the-counter options were actually written. The natural inputs are the stock price, the
exercise price, the time to maturity, the prevailing short-term interest rate, and an
estimate of the stock's volatility from historical returns. The natural diagnostics are
whether a hedged position formed from a candidate slope rule earns a return uncorrelated
with the market, and whether buying options a candidate valuation marks cheap and selling
those it marks dear is profitable before transaction costs. Listed-option markets of the
kind that would make continuous rehedging cheap do not yet exist; the over-the-counter
market is thin and its transaction costs are large.

## Code framework

A pricing worksheet can already compute the terminal payoff, describe the lognormal
terminal stock distribution under an assumed drift, and discount a future amount at a
chosen rate.

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
