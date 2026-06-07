# Ski-rental: the randomized e/(e−1)-competitive rent-or-buy algorithm

## Problem

A recurring-vs-lump-sum decision under an unknown, adversarial horizon. Scale so renting costs 1 per day and buying costs B; once bought, no more rent. The horizon x (number of days) is revealed only as it unfolds. The clairvoyant pays C_OPT(x) = min(x, B). The same object models snoopy-cache coherence (1 per broadcast write vs. penalty B to invalidate) and lock spin-vs-block (cost ∝ spin time vs. fixed block cost B). Goal: an on-line rule minimizing the worst-case ratio (expected cost)/OPT.

## Key idea

A pure strategy is a buy-day cutoff A_a (rent a days, buy on day a+1), with cost x if x ≤ a and a+B otherwise. Equivalently, a buy day i means rent i−1 days and buy if the horizon reaches day i.

- **Deterministic optimum is the break-even rule** a = B−1 (rent until cumulative rent equals the purchase price): worst-case ratio 2 − 1/B, and no deterministic rule beats it. The oblivious adversary defeats any deterministic rule by ending the horizon the day after the known buy-day.
- **Randomize the buy-day** to hide it from the oblivious adversary. Choose buy-day i ∈ {1,…,B} with probability π_i. Forcing the per-horizon expected ratio to be equal across all horizons (so the adversary has no single worst horizon) yields a geometric distribution growing toward the deadline, and normalization fixes the probabilities:

  π_i = α · r^{i−1},  r = B/(B−1) = 1/(1 − 1/B),  α = 1 / ((B−1)(r^B − 1)).

  The competitive factor is ρ = 1 + (B−1)α = r^B / (r^B − 1) = 1 / (1 − (1 − 1/B)^B), which increases to **e/(e−1) ≈ 1.58** as B → ∞.
- **Optimality (lower bound).** By Yao's principle, take the adversary horizon distribution with Pr[X ≥ t] = (1 − 1/B)^{t−1}. Every deterministic cutoff then has the same expected cost B, while OPT has expected cost B(1 − (1 − 1/B)^B), so every randomized algorithm has ratio ≥ 1 / (1 − (1 − 1/B)^B) → e/(e−1). Upper and lower bounds are the same geometric object viewed from algorithm and adversary sides.
- **Continuous (spin-block) version.** Buy-time z drawn on [0,1] (scale B=1) with density p(z) = e^z/(e−1); the geometric becomes its exponential limit and the competitive factor is exactly e/(e−1).

## Algorithm

1. Compute r = B/(B−1), α = 1/((B−1)(r^B − 1)), and π_i = α r^{i−1} for i = 1..B.
2. Sample a buy-day i ∼ π once, before play.
3. Rent through day i−1; on day i, buy if the horizon reaches it. Expected cost ≤ (e/(e−1))·OPT on every horizon.

Continuous: sample z = ln(1 + U(e−1)) with U ∼ Uniform[0,1]; spin until time z, then block.

## Code

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

Running this prints the deterministic ratio sliding to 2 and the randomized worst-horizon ratio sliding to e/(e−1) ≈ 1.5820 (1.4874, 1.5353, 1.5774, 1.5815 for B = 5, 10, 100, 1000).
