The problem I want to settle is a recurring-versus-lump-sum decision made under a horizon I cannot see. The same object wears three costumes: a skier renting at $1$ per day who can instead buy for a one-time cost $B$ and end all future rent; a snoopy cache paying $1$ per broadcast write to keep a shared block coherent, or paying a penalty $B$ once to invalidate it and stop the broadcasts; and a process spinning on a lock at cost proportional to the time it waits, or blocking once at a fixed context-switch cost $B$. Strip the dressing and there is one object: a recurring cost of $1$ per day, a lump sum $B$ that abolishes the recurring cost forever, and a horizon $x$ — the number of days I will actually need the thing — that is hidden when I must act and may be chosen by an adversary to punish whatever I decide. I scale rent to $1$/day and buy to $B$ so they are comparable, and I measure myself not against an average over some assumed distribution of $x$ but against the clairvoyant who knows the horizon. That clairvoyant pays $C_{\mathrm{OPT}}(x) = \min(x, B)$: rent the whole time if $x < B$, buy on day one if $x \ge B$. This is the competitive-analysis lens of Sleator and Tarjan — an algorithm is $c$-competitive if its cost is at most $c\cdot C_{\mathrm{OPT}}$ plus a constant on every input — and I want the smallest possible $c$.

The natural on-line strategies form a one-parameter family. At any moment I have been renting; a pure strategy $A_a$ rents through day $a$ and buys on day $a+1$, costing $x$ if $x \le a$ (the season ended before I bought) and $a+B$ if $x > a$ (I rented $a$ days, then bought). Because I am deterministic, an oblivious adversary can simulate me, knows exactly which day I buy, and ends the horizon the instant after: it sets $x = a+1$, so I pay $a+B$ against $\mathrm{OPT} = \min(a+1, B)$. Minimizing $(a+B)/\min(a+1,B)$ pulls $a$ to $B-1$ from both directions — I want the denominator at least $B$ (so $a \ge B-1$) and the numerator small (so $a$ small) — giving the break-even rule: rent until the rent I have spent equals the purchase price, then buy. It pays $2B-1$ against $B$, a ratio of $2 - 1/B$, and no deterministic cutoff does better. The leak is structural: I commit to a single buy-day, and the adversary aims at it. Always-buy and always-rent are worse still, each unbounded on one side, so the cutoff is the only sensible degree of freedom and the whole family stalls just under $2$.

The fix is to plug the leak by randomizing the buy-day so the oblivious adversary, which sees my code and my distribution but not my coin flips, can no longer aim at the day after I buy. I propose the randomized $e/(e{-}1)$-competitive rent-or-buy rule: draw the buy-day $i \in \{1,\dots,B\}$ from a distribution $\pi$ once before play, rent through day $i-1$, and buy on day $i$ if the horizon reaches it. To find the right $\pi$ I write the expected cost at a fixed horizon $k$. A buy on day $i$ costs $(i-1)+B$ when $i \le k$ (the season reaches the buy day) and costs $k$ when $i > k$ (the season ends first), so $E[C(k)] = \sum_{i\le k}\pi_i(B+i-1) + k\sum_{i>k}\pi_i$. The design principle is to leave the adversary no single worst horizon: I force the expected ratio to be flat, $E[C(k)] = \rho k$ for every $k \le B$, where $\mathrm{OPT}(k)=k$. This is one equation per horizon, a whole system that pins $\pi$ down. Looking at the increment from $k$ to $k+1$, the term for $i=k+1$ flips out of the rent-only bucket — it contributed $\pi_{k+1}k$ and now contributes $\pi_{k+1}(B+k)$, a change of $B\pi_{k+1}$ — while every later term's multiplier rises by one, adding the tail $T_{k+1} = \sum_{i>k+1}\pi_i$. So the flattened constraint gives $B\pi_{k+1} + T_{k+1} = \rho$. Differencing two consecutive such equations and using $T_k = \pi_{k+1}+T_{k+1}$ yields $\pi_{k+1}(B-1) = B\pi_k$, hence

$$\pi_{k+1} = \pi_k\,\frac{B}{B-1}, \qquad \pi_i = \alpha\, r^{\,i-1}, \quad r = \frac{B}{B-1} = \frac{1}{1 - 1/B} > 1.$$

The probabilities are geometric and *grow* toward the deadline — exactly right, since the longer I have rented without the season ending, the more willing I should be to commit — and the mass piles up near day $B$. Two boundary facts close it. First, the support must stop at day $B$: once I have rented $B$ days the long-horizon $\mathrm{OPT}$ is already $B$, so delaying the purchase only adds wasted rent. Second, normalization fixes the constant. Since $r-1 = 1/(B-1)$,

$$\sum_{i=1}^{B}\pi_i = \alpha\,\frac{r^B-1}{r-1} = \alpha(B-1)(r^B-1) = 1 \;\Longrightarrow\; \alpha = \frac{1}{(B-1)(r^B-1)}.$$

The competitive factor falls out of the day-one boundary equation $\rho = 1 + (B-1)\alpha$. Substituting $\alpha$,

$$\rho = 1 + \frac{1}{r^B - 1} = \frac{r^B}{r^B - 1} = \frac{1}{1 - (1 - 1/B)^B} \;\xrightarrow[B\to\infty]{}\; \frac{e}{e-1} \approx 1.58,$$

because $(1-1/B)^B \to 1/e$. I did not choose $1.58$; it is forced by "make every horizon equally bad and let the probabilities sum to one." For $B = 5, 10, 100$ the factor is $\approx 1.487, 1.535, 1.577$, climbing to $1.582$, and it beats $2 - 1/B$ at every $B \ge 2$. Horizons $k > B$ cannot do worse, since there $\mathrm{OPT}(k) = B$ and my expected cost equals $E[C(B)] = \rho B$, already equalized.

This is exactly optimal, not merely good, and the proof is Yao's principle: lower-bound any randomized algorithm by exhibiting a horizon distribution $X$ on which the best deterministic cutoff is bad on average. Demanding that all cutoffs tie forces the reciprocal geometry on the tail — making $A_a$ and $A_{a-1}$ equal gives $\Pr[X \ge a+1] = (1-1/B)\Pr[X \ge a]$, so $\Pr[X \ge t] = (1-1/B)^{t-1}$. Under this $X$ every deterministic cutoff costs exactly $B$, while $\mathrm{OPT}$ costs $\sum_{t=1}^{B}\Pr[X \ge t] = B(1-(1-1/B)^B)$, so every randomized algorithm has ratio at least $1/(1-(1-1/B)^B) \to e/(e-1)$. Upper and lower bounds are the same geometric object seen from the two sides: the algorithm spreads buy-days with factor $r = 1/(1-1/B)$, the adversary spreads horizons with the reciprocal $1-1/B$, and they meet at $1/(1-(1-1/B)^B)$. The continuous spin-block version, where waiting time is real, scales $B=1$ and replaces $\pi$ by a density $p(z)$ on $[0,1]$. Flattening $E[C(\tau)] = \int_0^\tau p(z)(z+1)\,dz + \tau\int_\tau^1 p(z)\,dz = \rho\tau$ and differentiating twice gives $p' = p$, so $p(z) = Ce^z$; normalizing $\int_0^1 Ce^z\,dz = C(e-1) = 1$ gives $p(z) = e^z/(e-1)$, the exponential limit of the discrete geometric, and the boundary value at $\tau = 1$ makes the factor $\rho = p(1) = e/(e-1)$ exactly — the truncation at $z=1$ matters for the same waste reason as before.

```python
import math, random

def opt_cost(x, B):
    # clairvoyant: rent everything if the horizon is short, else buy on day one
    return min(x, B)

def det_cost(buy_day, x, B):
    # buy_day is 1-indexed: rent buy_day-1 days, then buy if the horizon reaches it
    return x if x < buy_day else (buy_day - 1) + B

def best_det_ratio(B):
    # break-even buy day B: worst horizon x = B gives (2B-1)/B = 2 - 1/B
    return det_cost(B, B, B) / opt_cost(B, B)

def buy_day_distribution(B):
    # pi_i = alpha * r^(i-1), r = B/(B-1), i=1..B.
    # alpha normalizes the probabilities; rho = 1 + (B-1)*alpha is the ratio.
    if B == 1:
        return [1.0]
    r = B / (B - 1)
    alpha = 1.0 / ((B - 1) * (r ** B - 1.0))
    return [alpha * r ** (i - 1) for i in range(1, B + 1)]

def expected_ratio_exact(B):
    # worst over horizons of E_i[det_cost(i, x, B)] / opt_cost(x, B)
    pi = buy_day_distribution(B)
    return max(
        sum(pi[i - 1] * det_cost(i, x, B) for i in range(1, B + 1)) / opt_cost(x, B)
        for x in range(1, 2 * B + 1)
    )

def sample_buy_day(pi, rng):
    u, c = rng.random(), 0.0
    for i, p in enumerate(pi, start=1):
        c += p
        if u <= c:
            return i
    return len(pi)

def adversary_tail(t, B):
    # Yao lower-bound distribution: Pr[X >= t] = (1 - 1/B)^(t-1)
    return (1.0 - 1.0 / B) ** (t - 1)

def continuous_density(z):
    # spin-block density p(z) = e^z/(e-1) on [0,1]
    if 0.0 <= z <= 1.0:
        return math.exp(z) / (math.e - 1.0)
    return 0.0

def sample_buy_time_continuous(rng):
    # inverse CDF F(z) = (e^z - 1)/(e - 1), so z = ln(1 + u(e - 1))
    return math.log(1.0 + rng.random() * (math.e - 1.0))

def continuous_ratio(z, d):
    # scale buy cost to 1; spin until z, then block if the lock is still held
    cost = d if d < z else z + 1.0
    return cost / min(d, 1.0)

if __name__ == "__main__":
    for B in [5, 10, 100, 1000]:
        print(B, round(best_det_ratio(B), 4), round(expected_ratio_exact(B), 4))
    print("e/(e-1) =", math.e / (math.e - 1))
```
