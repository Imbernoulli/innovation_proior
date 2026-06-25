OK, let me think this through from scratch. I want the value, today, of a call option — the right to buy a share at a fixed exercise price `c` on a fixed future date `t*`. The payoff on that date is not in doubt: if the stock finishes at `x_T` the option is worth `max(x_T − c, 0)`, nothing if it lands below the strike and the in-the-money amount otherwise. Everything hard is in the words "today." The stock will wander between now and `t*`, so the present value has to be some kind of average over where it might end up.

The recipe everyone reaches for is the obvious one: take the expected payoff at maturity and discount it back. So I need two numbers. First, to compute the expected payoff I need to know how fast the stock drifts on average — its expected rate of return, call it `α`. Second, to bring that expected payoff back to the present I need a discount rate appropriate to a claim as risky as the option. And here is where it falls apart, and it falls apart twice. The expected return `α` is exactly the quantity nobody can pin down — it is the most argued-over number in all of finance. And the discount rate is worse than unknown, it is incoherent: the option is a leveraged bet on the stock, so its riskiness is not a fixed number. Deep in-the-money the option moves nearly dollar-for-dollar with the share and behaves almost like the stock; far out-of-the-money it barely twitches when the stock moves a dollar, yet in percentage terms it is wildly more volatile than the stock. The leverage, and therefore the option's risk, changes with the stock price and with how much time is left. So there is no single discount rate; the right rate depends on `x` and `t`, the very things I am trying to solve for. The honest recipe eats its own tail.

Look at what the best existing attempts do with these two holes, because they map the trap precisely. Sprenkle, back in 1961, assumes the terminal stock price is lognormal — which is the right distributional instinct, I'll come back to it — and writes the option value as the expected cut-off payoff:
```
k·x·N(b₁) − k*·c·N(b₂),
```
with `b₁, b₂` the usual log-moneyness terms over `v·√(t*−t)`. That form is gorgeous, it is almost the answer. But `k` is the ratio of the stock's expected terminal price to today's price — that is just the unknown `α` wearing a hat — and `k*` is a discount factor tied to the option's risk. Sprenkle tries to estimate `k` and `k*` from data and frankly admits he can't. Boness in 1964 patches the discount hole by discounting at the stock's own expected return; but that quietly assumes the option is exactly as risky per dollar as the stock, which the leverage argument says is false, and it still needs `α`. Samuelson's 1965 rational-warrant-pricing paper is the most careful: warrant value as a function of the stock price, lognormal terminal value, expected payoff discounted — but at a warrant rate `β` distinct from the stock rate `α`, and he says outright that there is no equilibrium theory telling you what `β` should be. He and Merton in 1969 go further and admit that discounting the cut-off expected value at an exogenous rate just isn't a legitimate procedure, and that the discount rates ought to be tied to investors holding both the stock and the option — but their fix runs through the representative investor's utility function, so the price ends up depending on the assumed shape of that utility. Every one of these carries an unobservable `α`, or an unjustified `β`, or a utility function. Nobody prices the option from things you can actually measure.

So I won't discount an expected terminal value. Let me follow Treynor's habit instead and write the option value as a function of the things it actually depends on: `w(x, t)`, the value when the stock is at `x` with time `t`. Treynor taught me this when I was at Arthur D. Little — he had a "value equation," a differential equation for the present value of a firm's uncertain cash flows, derived by writing value as a function of the state and applying the capital asset pricing model moment by moment. I remember his equation had an error: he had dropped some second-derivative terms, and it didn't come right until we put them back. Hold onto that — the second-derivative terms are not optional. Writing `w(x,t)` and demanding it be consistent with equilibrium at every instant feels far more promising than guessing a discount rate, because it lets the structure of the problem, rather than an arbitrary parameter, determine the answer.

I need the stock's dynamics. Bachelier modeled the price itself as Brownian motion, but that's wrong for a stock: it gives the same absolute jitter at \$5 and at \$500, and it lets the price go negative. Samuelson's fix is the one I'll use — the logarithm of the price is Brownian motion. Then percentage returns have constant variance, the price stays positive, and the terminal price is lognormal. Over a short interval `Δt`, the change `Δx` has expected value `α·x·Δt` and variance `x²·v²·Δt`, where `v²` is the variance rate of the return, taken constant.

Now expand `w(x, t)` over a short step. Naively I'd write `Δw = w₁·Δx + w₂·Δt` and stop. But this is precisely Treynor's mistake. `Δx` is random with standard deviation of order `√Δt`, so the accumulated quadratic variation of `Δx` is of order `Δt` — the same order as the time step itself. I cannot drop the second-order term in `x`; it survives into the leading order. So:
```
Δw = w₁·Δx + w₂·Δt + ½·w₁₁·Δx²,
```
subscripts being partial derivatives in the stock price (first argument) and time (second). In the continuous-time limit the quadratic-variation term contributes `x²v²·Δt`, so `½·w₁₁·Δx²` becomes the finite `½·w₁₁·x²v²·Δt` term. That single term, the one ordinary calculus throws away, is going to carry everything.

Let me apply the capital asset pricing model to the warrant the way Treynor applied it to cash flows. The model says a security's expected excess return is proportional to its beta — its covariance with the market over the market's variance — and that only non-diversifiable risk is rewarded. Call the stock's beta `βx`, and call the market reward per unit beta `λ`. Then the stock's expected proportional move over `Δt` is
```
E(Δx/x) = r·Δt + λ·βx·Δt.
```
The warrant's market covariance comes through the stock move, and a one-dollar stock move changes the warrant by `w₁` dollars, so the warrant beta is `(x·w₁/w)·βx`. CAPM therefore says
```
E(Δw) = r·w·Δt + λ·x·w₁·βx·Δt.
```
But Itô says the same expected warrant change is
```
E(Δw) = w₁·E(Δx) + ½·w₁₁·v²·x²·Δt + w₂·Δt
      = r·x·w₁·Δt + λ·x·w₁·βx·Δt + ½·w₁₁·v²·x²·Δt + w₂·Δt.
```
Equating the two expressions cancels the `λ·x·w₁·βx` risk-premium term and leaves
```
w₂ = r·w − r·x·w₁ − ½·v²·x²·w₁₁.
```
And here is the thing that stops me cold when I stare at the equation. The stock's expected return `α` is gone. It has dropped out. The warrant value does not depend on how fast the stock is expected to rise. More than that — it does not depend on how the stock's risk splits into the part you can diversify away and the part you can't. It depends only on the *total* risk of the stock, the variance rate `v²`. That fascinates me, and it also unsettles me, because I came in believing the expected return had to matter — surely a warrant on a stock that's expected to soar is worth more than one on a stock expected to stagnate? But the equation says no. Whatever is going on, the drift is irrelevant and only the volatility counts.

The equation has, I'm fairly sure, a unique solution once I impose the terminal condition `w(x, t*) = max(x − c, 0)` and one more boundary condition that, honestly, I don't fully understand yet. So I sit down to solve it. And I cannot. I have a doctorate in applied mathematics and a physics degree, and I spend days on this differential equation and get nowhere. I do not recognize what class of equation I am staring at, so I am stuck. I set the warrant problem aside.

Now Myron and I are working together, and the fascination with the vanishing `α` turns into a lever. If the value genuinely does not depend on the stock's expected return, then I am *free to choose* the expected return — any value I like — because the answer can't depend on it. That is a strange and powerful liberty. So let me pick the most convenient value imaginable: let the stock's expected return equal the riskless interest rate `r`. In the language of the capital asset pricing model, I am pretending the stock has zero beta — all of its risk is diversifiable, none of it is the kind the market pays a premium for. That is surely false about the real stock, but it doesn't matter, because the warrant value is the same for every assumed expected return, so it must equal the value computed under this fictitious one.

What does this choice buy me? Trace the consequences. In a world where the stock has zero beta, with drift `r` and constant volatility, the terminal stock price is lognormal with a known mean, so the expected cut-off payoff is computable — it is exactly Sprenkle's expression, but now with the stock's expected appreciation set to `r` rather than left as the mystery `k`. That handles the first hole. The second hole was the discount rate, and here the zero-beta choice does something I didn't anticipate when I made it for mere convenience. If the stock has zero beta, then the option — being just a function of the stock — also has zero beta: all of the stock's risk is diversifiable, so all of the option's risk is diversifiable too. A zero-beta security must, in equilibrium, earn exactly the riskless rate, because none of its risk is rewarded. So the option's expected return is `r`, at every instant, regardless of the stock price or the time. The discount rate that takes the option's expected future value back to the present is the *constant* `r` — not a function of `x` and `t` after all, once I'm in the zero-beta world. The thing that was incoherent in general — a single discount rate for a claim of varying risk — becomes coherent and constant because I chose the drift that zeros out the beta. So both holes close under the same choice; that's why it's the one to make.

So: take Sprenkle's formula for the expected terminal value, put `r` in for the stock's expected return, and discount the whole thing back at the constant `r`. That collapses Sprenkle's free `k` and `k*` to determined values, and out comes a single formula relating the option value to the stock price, the strike, the time, the rate, and the volatility — nothing else. The candidate is `w = x·N(d₁) − c·e^{−r(t*−t)}·N(d₂)` with the `d`'s I'll pin down below.

This came from an expected-value-and-discount argument, not from the differential equation, so I have no right yet to claim the two agree. I owe myself the check: does this `w` actually satisfy `w₂ = rw − rxw₁ − ½v²x²w₁₁`, the equation I couldn't solve in 1969? Analytically the substitution is a pile of chain-rule bookkeeping; let me instead pin one concrete point numerically so I'm not fooling myself. Take `c=100`, `r=0.05`, `v=0.20`, a year to maturity, and ask at `x=100`, half a year out. Differencing the formula gives `w₁ ≈ 0.5977`, `w₁₁ ≈ 0.02736`, `w₂ ≈ −8.1160`, and `w ≈ 6.8887`. The right-hand side is `0.05·6.8887 − 0.05·100·0.5977 − ½·0.04·100²·0.02736 = 0.3444 − 2.9887 − 5.4717 = −8.1160`. The two sides match to within `3×10⁻⁶` — that gap is just the finite-difference noise, not a real discrepancy. So the expected-value formula and the 1969 equation really are the same object; the convenient drift choice didn't smuggle in an error. And at `t=t*` the `√(t*−t)` denominator sends `d₁,d₂` to `±∞`, collapsing `w` to `max(x−c,0)`, so the terminal condition holds too.

I want to be more honest about *why* the drift cancels, because the zero-beta story, while it works, leans on the whole apparatus of equilibrium and the capital asset pricing model, and Merton is pushing me on whether that apparatus is really needed or even airtight as the trading interval shrinks. Let me try to get the same answer from no-arbitrage alone, with no model of the market at all. Suppose I form a portfolio: long one share of stock, and short some number `n` of options against it. The value of this position is `x − n·w`. Over a short step, the stock changes by `Δx` and the option by `Δw = w₁·Δx + (w₂ + ½·w₁₁·x²v²)·Δt`. So the position changes by
```
Δx − n·Δw = (1 − n·w₁)·Δx − n·(w₂ + ½·w₁₁·x²v²)·Δt.
```
The randomness lives entirely in the `Δx` term. If I choose `n = 1/w₁` — short exactly `1/w₁` options for each share, equivalently hold `w₁` shares for each option — the coefficient `(1 − n·w₁)` becomes zero and the `Δx` term disappears. What's left,
```
−(1/w₁)·(w₂ + ½·w₁₁·x²v²)·Δt,
```
has no `Δx` in it at all. It is deterministic over the step. The hedged position is riskless.

Stare at that. The single thing that made the option impossible to price — its dependence on the unknowable drift `α` — lived in `Δx`, because `Δx` is where the stock's expected move sits. By cancelling `Δx`, the hedge cancels `α` along with it. The expected return never had a chance to enter. This is why it dropped out of my differential equation back in 1969: the equation is secretly the statement that a `Δx`-neutral position is left holding only `Δt` terms.

Now the no-arbitrage hammer. A position that is riskless over the step must earn the riskless rate `r` over the step — otherwise I could borrow at `r`, build the position, and harvest a sure profit, and so could everyone, and the prices would move until the free lunch closed. The riskless growth on the equity `(x − w/w₁)` over `Δt` is `(x − w/w₁)·r·Δt`. Set the deterministic change equal to that:
```
−(½·w₁₁·v²·x² + w₂)·Δt / w₁ = (x − w/w₁)·r·Δt.
```
Drop the `Δt`, multiply through by `w₁`, and rearrange:
```
w₂ = r·w − r·x·w₁ − ½·v²·x²·w₁₁.
```
There is the differential equation, derived now from nothing but the absence of arbitrage — no capital asset pricing model, no utility function, no equilibrium assumption beyond "you can't make a riskless profit." Only `r` and `v²` appear; `α` is absent because the hedge annihilated it. The terminal condition is the payoff, `w(x, t*) = max(x − c, 0)`.

One caveat I owe myself. The hedge ratio `w₁` is not constant — `w₁` itself depends on `x` and `t`, so as the stock moves I must keep rebalancing the number of shares per option. Over a *finite* interval the position is not exactly riskless, because the curvature `w₁₁` means `Δw` is not exactly `w₁·Δx`. The cancellation of `Δx` is exact only in the limit of continuous rebalancing. Merton's sharpening is exactly this: if I trade continuously inside the model, the hedge is exactly riskless, the replication is exact, and the no-arbitrage conclusion is rigorous rather than approximate. And if I don't rebalance continuously, the residual risk is the second-order curvature risk, which has zero covariance with the market — so it is purely diversifiable and earns no premium anyway. Either way the conclusion holds: the hedged position earns `r`. The continuous-trading version is the cleaner one because it needs only arbitrage, so that is the derivation to lead with; the zero-beta version is the scaffolding that first revealed why `α` doesn't matter.

Now solve the equation. It's the same one I couldn't crack alone, so I keep the constants visible instead of trusting my eye. Let
```
a = r − ½v²,
A = 2a/v²,
w(x, t) = e^{r(t − t*)} · y(u, s),
u = A · [ ln(x/c) − a·(t − t*) ],
s = −A·a·(t − t*).
```
The exponential strips out the discounting factor, `u` is log-price with the drift `a = r − ½v²` folded in, and `s` is the rescaled time-to-maturity. If `a=0`, I take the limiting version of the same change of variables; a coordinate singularity cannot make the option price jump. Now the signs have to be exact. Holding `t` fixed,
```
x·w₁ = e^{r(t − t*)} · A·y₁,
x²·w₁₁ = e^{r(t − t*)} · (A²·y₁₁ − A·y₁),
w₂ = e^{r(t − t*)} · (r·y − A·a·y₁ − A·a·y₂),
```
where the subscripts on `y` are now derivatives in `u` and `s`. Substitute these into
`w₂ = r·w − r·x·w₁ − ½v²x²w₁₁` and divide out the exponential:
```
r·y − A·a·y₁ − A·a·y₂
= r·y − r·A·y₁ − ½v²·(A²·y₁₁ − A·y₁)
= r·y − A·(r − ½v²)·y₁ − ½v²A²·y₁₁.
```
Since `a = r − ½v²` and `A = 2a/v²`, the `r·y` terms cancel, the `A·a·y₁` terms cancel, and `½v²A² = A·a`. What remains is exactly
```
y₂ = y₁₁,
```
the heat equation. This is what I failed to recognize three years ago: my warrant equation is the equation for heat diffusing along a rod, whose solution has been written down since the nineteenth century. At maturity `t=t*`, `s=0` and `x=c·exp(u/A)`, so the terminal profile is the cut-off exponential
```
y(u, 0) = max(c·[exp(u/A) − 1], 0).
```
The heat equation says to convolve this profile with a Gaussian kernel. Written back in stock language, that same Gaussian average is cleaner:
```
τ = t* − t,
X_T = x·exp((r − ½v²)τ + v√τ·Z),     Z ~ N(0,1),
w(x,t) = e^{−rτ}·E[(X_T − c)^+].
```
This is not a new assumption; it is the heat-kernel integral after the change of variables is undone. The exercise event is
```
X_T > c
⇔ Z > [ln(c/x) − (r − ½v²)τ] / (v√τ)
⇔ Z > −d₂,
```
where
```
d₂ = [ ln(x/c) + (r − ½v²)τ ] / (v√τ).
```
So the strike term contributes `c·e^{−rτ}·N(d₂)`. The stock term needs one square completed:
```
E[X_T·1_{X_T>c}]
= x·e^{(r−½v²)τ} ∫_{−d₂}^{∞} e^{v√τ z} φ(z) dz
= x·e^{rτ} ∫_{−d₂−v√τ}^{∞} φ(y) dy
= x·e^{rτ}·N(d₂ + v√τ),
```
with `φ` the standard normal density. Let `d₁ = d₂ + v√τ`. Discounting that stock term back by `e^{−rτ}` and subtracting the discounted strike term gives
```
w(x, t) = x·N(d₁) − c·e^{−rτ}·N(d₂),
d₁ = [ ln(x/c) + (r + ½v²)τ ] / (v√τ),
d₂ = [ ln(x/c) + (r − ½v²)τ ] / (v√τ) = d₁ − v√τ,
```
with `N` the standard normal cumulative distribution. The discount factor is the riskless rate over the remaining life — which is right, because in the hedged world the option earns `r`.

Let me read the formula back and check it against everything I know about how an option should behave. The expected return `α` does not appear — exactly as it must, since the hedge cancelled it; the value depends on the stock price, the strike, the time, the rate `r`, and the volatility `v`, all observable. Deep in-the-money, `ln(x/c)` is large and positive, both `N`'s go to 1, and `w → x − c·e^{−r(t*−t)}`: the option becomes the stock minus the present value of the strike, i.e. you're as good as committed to buying at `c`, so you hold the share less a riskless bond for the strike. Deep out-of-the-money, both `N`'s go to 0 and `w → 0`. At maturity `t = t*`, the `√(t*−t)` in the denominator drives `d₁, d₂` to `±∞` according to the sign of `ln(x/c)`, so `w → x − c` for `x > c` and `0` for `x < c` — the payoff. As volatility becomes very large, `d₁` goes to `+∞` and `d₂` goes to `−∞`, so the formula approaches the ceiling `x`; Merton's monotonicity result says it moves upward toward that ceiling as the variance rate increases.

And the two `d`'s differ by exactly `v·√(t*−t)`, the standard deviation of the log return over the remaining life. The `± ½v²` is the lognormal mean-correction from Itô — the same second-order term that I couldn't drop in the expansion, now showing up as the gap between the term that carries the stock (`d₁`, with `+½v²`) and the term that carries the probability of finishing in the money (`d₂`, with `−½v²`). The whole edifice traces back to that one surviving second-derivative term.

The hedge ratio falls out too. The number of shares I must hold per option to stay riskless is `w₁`. When I differentiate the formula, the raw derivative is
```
w₁ = N(d₁) + φ(d₁)/(v√τ) − c·e^{−rτ}·φ(d₂)/(x·v√τ),
```
where `φ` is the standard normal density. But the two density terms are the same dollar amount in disguise:
```
x·φ(d₁) = c·e^{−rτ}·φ(d₂),
```
because `d₁ = d₂ + v√τ` and the exponent difference exactly reproduces `ln(x/c) + rτ`. They cancel, leaving
```
w₁ = N(d₁).
```
That is a clean, computable quantity between 0 and 1 that tells the hedger, instant by instant, how many shares to hold against each option. It moves from near 0 when the option is deep out-of-the-money to near 1 when it's deep in, and I must rebalance to it continuously to keep the position riskless. That `w₁ = N(d₁)` was the missing piece I needed in 1969 and couldn't get without the closed form: to run the zero-beta hedge I had to know how many shares belong on the stock side per option, and only the solved formula tells me.

Puts come for free: a put is the right to sell at `c`, with terminal payoff `max(c − x, 0)`, and the same hedging argument with the sign of the position flipped gives the put formula. The cleaner check is parity. A long call and short put with the same strike and maturity pays `X_T − c` in every state, so today it must be worth `x − c·e^{−rτ}`. Therefore `p = w − x + c·e^{−rτ}`; using `N(−d)=1−N(d)` turns that into `p(x,t)=c·e^{−rτ}·N(−d₂) − x·N(−d₁)`.

Let me fill the worksheet now, keeping the older payoff, lognormal, and discounting primitives in place while the empty valuation rule becomes the formula and the hedge rule becomes `N(d₁)`.

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
    """European call value. Alpha and discount_rate are deliberately unused:
    the riskless hedge removes the stock drift and forces discounting at r."""
    if tau <= 0:
        return call_payoff(x, c)
    # d1 carries the +1/2 sigma^2 Ito mean-correction; norm.cdf(d2) is the
    # risk-neutral probability of finishing in the money.
    d1 = (np.log(x / c) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    # Discount at the riskless rate, because in the hedged world the option earns r.
    return x * norm.cdf(d1) - discount(c, r, tau) * norm.cdf(d2)

def put_value(x, c, tau, r, sigma):
    if tau <= 0:
        return max(c - x, 0.0)
    d1 = (np.log(x / c) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return discount(c, r, tau) * norm.cdf(-d2) - x * norm.cdf(-d1)

def hedge_ratio(x, c, tau, r, sigma):
    """Shares to hold per call to stay riskless: w_1 = N(d1).
    Rebalance to this continuously as x and tau change."""
    d1 = (np.log(x / c) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)

if __name__ == "__main__":
    x, c, tau, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
    print("call :", round(call_value(x, c, tau, r, sigma), 4))
    print("put  :", round(put_value(x, c, tau, r, sigma), 4))
    print("delta:", round(hedge_ratio(x, c, tau, r, sigma), 4))
    # Put-call parity sanity check: C - P should equal x - c*exp(-r*tau).
    lhs = call_value(x, c, tau, r, sigma) - put_value(x, c, tau, r, sigma)
    print("parity gap:", round(lhs - (x - c * np.exp(-r * tau)), 10))
```

The causal chain, start to finish: pricing an option looks like it needs the stock's expected return (unknown) and a discount rate for a claim whose risk varies with price and time (incoherent). Write the option value as a function `w(x,t)` and expand it over a short step; the second-derivative term `½w₁₁x²v²` survives because the stock's increment is order `√Δt`. Short `1/w₁` options against one share and the random `Δx` term cancels — and with it the drift `α`, since `α` lived only in `Δx`. A position with no `Δx` is riskless, so by no-arbitrage it earns the riskless rate `r`; equating its deterministic change to `r` times its value gives the equation `w₂ = rw − rxw₁ − ½v²x²w₁₁`, with only `r` and `v²` in it. Continuous rebalancing makes the hedge exactly riskless (Merton); discrete rebalancing leaves only diversifiable curvature risk, which earns no premium, so the conclusion stands. Change variables to reach the heat equation `y₂ = y₁₁`, convolve the cut-off-lognormal initial condition with the Gaussian kernel, and out comes `w = x·N(d₁) − c·e^{−r(t*−t)}·N(d₂)`, with hedge ratio `N(d₁)` — a price and a hedging rule built entirely from observables.
