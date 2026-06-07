# The Black–Scholes–Merton Option-Pricing Formula

## Problem

Value a European call today: the right to buy one share at exercise price `c` on maturity
date `t*`, when the stock trades at `x` with `τ = t*−t` years remaining. The terminal payoff
is `max(x−c, 0)`. The obstacle to valuing it earlier than maturity is that the naive
"discount the expected payoff" recipe needs two quantities it cannot supply: the stock's
expected return `α` (unobservable) and a discount rate for a claim whose risk varies with the
stock price and time (no single rate is correct).

## Key idea

Model the stock as geometric Brownian motion: over a short interval `Δt`, `Δx` has mean
`α·x·Δt` and variance `x²σ²·Δt`. Hold one share long and short `1/w₁` options, where
`w₁ = ∂w/∂x` is the option's sensitivity to the stock. To first order one option moves
`w₁·Δx`, so this position has **no exposure to `Δx`** — it is riskless
over the interval (exactly so in the limit of continuous rebalancing). The random `Δx` term
is the only place the unknown drift `α` lived, so cancelling it cancels `α`. By no-arbitrage,
a riskless position must earn the riskless rate `r`. That single condition determines the
option value with no reference to `α` and no utility function — the value rests entirely on
observables: `x`, `c`, `τ`, `r`, and the volatility `σ`.

## The no-arbitrage hedging argument

With `Δw = w₁·Δx + (w₂ + ½·σ²x²·w₁₁)·Δt` (Itô: the second-order term survives because
`Δx² → σ²x²·Δt`), the hedged equity `x − w/w₁` changes by
```
−(½·σ²x²·w₁₁ + w₂)·Δt / w₁,
```
which is deterministic. Setting this equal to `r·(x − w/w₁)·Δt` (riskless ⇒ earns `r`) and
rearranging gives the partial differential equation.

## The Black–Scholes PDE

```
w₂ = r·w − r·x·w₁ − ½·σ²·x²·w₁₁,
```
with `w₁ = ∂w/∂x`, `w₁₁ = ∂²w/∂x²`, `w₂ = ∂w/∂t`, and terminal condition
`w(x, t*) = max(x − c, 0)`. Only `r` and `σ²` appear; the expected return is absent. The
heat-equation substitution used to solve it is
```
a = r − ½σ²,     A = 2a/σ²,
w(x,t) = e^{r(t−t*)}·y(u,s),
u = A·[ln(x/c) − a·(t−t*)],
s = −A·a·(t−t*).
```
The chain-rule identities are
```
x·w₁ = e^{r(t−t*)}·A·y₁,
x²·w₁₁ = e^{r(t−t*)}·(A²·y₁₁ − A·y₁),
w₂ = e^{r(t−t*)}·(r·y − A·a·y₁ − A·a·y₂).
```
Substituting these into the PDE cancels the `r·y` and `A·a·y₁` terms; since
`½σ²A² = A·a`, the remaining equation is `y₂ = y₁₁`. The terminal payoff becomes a
cut-off exponential initial profile, and convolution with the Gaussian heat kernel yields
the closed form.

## The closed-form formula

For a European call:
```
w(x, t) = x·N(d₁) − c·e^{−r·τ}·N(d₂),
d₁ = [ ln(x/c) + (r + ½σ²)·τ ] / ( σ·√τ ),
d₂ = d₁ − σ·√τ = [ ln(x/c) + (r − ½σ²)·τ ] / ( σ·√τ ),
```
where `N` is the standard normal CDF and `τ = t*−t`. The hedge ratio (shares per call to stay
riskless) is `w₁ = N(d₁)`; rebalance to it continuously. For a European put,
`p(x,t) = c·e^{−r·τ}·N(−d₂) − x·N(−d₁)`, consistent with put–call parity
`w − p = x − c·e^{−r·τ}`.

The `±½σ²` is the Itô lognormal mean-correction — the same surviving second-derivative term
that made the hedge necessary. `N(d₂)` is the risk-neutral probability of finishing in the
money; `N(d₁)` weights the stock leg. Differentiating the call formula leaves `N(d₁)`
because the two density terms cancel via `x·φ(d₁) = c·e^{−rτ}·φ(d₂)`.

## Code

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
    """European call. Alpha and discount_rate are deliberately unused:
    the riskless hedge removes the stock drift and forces discounting at r."""
    if tau <= 0:
        return call_payoff(x, c)
    d1 = (np.log(x / c) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return x * norm.cdf(d1) - discount(c, r, tau) * norm.cdf(d2)

def put_value(x, c, tau, r, sigma):
    if tau <= 0:
        return max(c - x, 0.0)
    d1 = (np.log(x / c) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return discount(c, r, tau) * norm.cdf(-d2) - x * norm.cdf(-d1)

def hedge_ratio(x, c, tau, r, sigma):
    """Shares per call to stay riskless: w_1 = N(d1)."""
    d1 = (np.log(x / c) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)

if __name__ == "__main__":
    x, c, tau, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
    print("call :", round(call_value(x, c, tau, r, sigma), 4))
    print("put  :", round(put_value(x, c, tau, r, sigma), 4))
    print("delta:", round(hedge_ratio(x, c, tau, r, sigma), 4))
    lhs = call_value(x, c, tau, r, sigma) - put_value(x, c, tau, r, sigma)
    print("parity gap:", round(lhs - (x - c * np.exp(-r * tau)), 10))
```

The expected return on the underlying never appears. Pricing reduces to building a
continuously rebalanced stock-plus-option hedge that is riskless, forcing it to earn `r`,
and solving the resulting heat equation for the value and its hedge ratio.
