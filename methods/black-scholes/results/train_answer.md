A European call option gives the right to buy one share of stock at a fixed exercise price c on a fixed maturity date, and its terminal payoff is simply max(x - c, 0). The hard part is valuing it before maturity, because the stock price will wander randomly between now and then. The obvious recipe is to take the expected payoff at maturity and discount it back, but that recipe needs two quantities it cannot supply. To compute the expected payoff you need the stock's expected rate of return alpha, which is famously unobservable and debated. To discount that expected payoff you need a discount rate appropriate to the option's risk, yet the option is a leveraged claim on the stock: deep in-the-money it behaves like the stock itself, far out-of-the-money it is much more volatile in percentage terms, and that leverage changes with the stock price and with time. So there is no single correct discount rate, and the naive approach eats its own tail.

Earlier work illustrates the trap precisely. Sprenkle assumed a lognormal terminal stock price and wrote the option value in a form that is almost the final answer, but his expression contained two free parameters encoding the unknown expected return and the option's risk-adjusted discount rate, and he admitted he could not estimate them. Boness improved the discounting by using the stock's own expected return, but that wrongly assumes the option is exactly as risky per dollar as the stock and still requires alpha. Samuelson kept a separate warrant discount rate beta as a free parameter and noted that no equilibrium theory justified any particular choice. Samuelson and Merton moved toward a partial differential equation for the option price, but their resolution leaned on a representative investor's utility function, so the answer was not preference-free. None of these approaches priced the option purely from observable quantities.

The new method is the Black-Scholes-Merton option-pricing formula. It begins by writing the option value as a function w(x, t) of the current stock price x and current time t, and it models the stock as geometric Brownian motion: over a short interval the proportional change in price has mean alpha times the interval and variance sigma squared times the interval, where sigma is the stock's volatility. Because the stock's increment is of order square-root of delta t, its squared contribution is of order delta t and cannot be dropped from the expansion of w. The correct short-interval change is delta w equals w1 times delta x plus w2 times delta t plus one-half w11 times delta x squared, where subscripts denote partial derivatives. In the continuous limit the quadratic variation term becomes one-half w11 times x squared times sigma squared times delta t.

The key move is to form a portfolio that is instantaneously riskless. Hold one share long and short one over w1 options, so that the value of the position is x minus w over w1. Since one option moves by approximately w1 times delta x to first order, the random delta x term in the position's change is cancelled exactly in the limit of continuous rebalancing. The expected return alpha lived only in delta x, so canceling delta x removes alpha from the problem entirely. What remains is a deterministic change, and a position that is riskless must earn the riskless interest rate r by the law of one price; otherwise an arbitrageur could borrow at r, build the position, and pocket a sure profit. Equating the deterministic change of the hedged position to r times its value yields the partial differential equation w2 equals r times w minus r times x times w1 minus one-half sigma squared times x squared times w11, with terminal condition w(x, maturity) equals max(x - c, 0). Only r and sigma appear; alpha is gone.

This PDE can be transformed into the heat equation by a suitable change of variables: set a equal to r minus one-half sigma squared, A equal to 2a over sigma squared, and define rescaled log-price and time variables from x, t, and c. The terminal payoff becomes a cut-off exponential initial profile, and convolving it with the Gaussian heat kernel produces the closed-form solution. Written back in stock-price language, the value of a European call is w(x, t) equals x times N(d1) minus c times e to the minus r tau times N(d2), where tau is the time to maturity, N is the standard normal cumulative distribution function, d1 equals the logarithm of x over c plus (r plus one-half sigma squared) times tau, all divided by sigma times square-root of tau, and d2 equals d1 minus sigma times square-root of tau. The hedge ratio, the number of shares to hold per call to stay riskless, is w1 equals N(d1); it varies between 0 and 1 and must be rebalanced continuously. For a European put, put-call parity gives p(x, t) equals c times e to the minus r tau times N(negative d2) minus x times N(negative d1).

The formula depends only on the stock price x, the strike c, the time to maturity tau, the riskless rate r, and the volatility sigma. The expected return alpha has disappeared because the hedge cancelled it, and the discount rate is pinned to the riskless rate because the hedged position is riskless. The plus and minus one-half sigma squared terms are the Ito lognormal mean-correction, the direct descendant of the second-derivative term that ordinary calculus would discard. In the deep in-the-money limit both normal probabilities approach 1 and the call is worth x minus the present value of the strike; far out-of-the-money it approaches 0; at maturity it reproduces the payoff exactly.

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
    """European call. alpha and discount_rate are deliberately unused:
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
