OK, the thing that actually hurts is the hidden stopping time. I have a block of memory shared between two caches. One processor keeps writing to it, and every write has to be broadcast on the bus so the other cache stays coherent — that's one bus cycle per write, a steady drip of cost. I could stop the drip by invalidating the block in the other cache, but invalidation isn't free: when the other processor eventually reads that block again, it has to be re-fetched, and that re-fetch costs me the whole block transfer, call it p cycles. So I'm paying 1 per write to keep sharing, or I pay a one-time p to stop. If only I knew how many more writes were coming before the other cache wants the block back, I'd know exactly what to do. I don't.

And it's not just caching. A process spinning on a lock pays per unit of time it spins; or it can block and eat a fixed context-switch cost C. Same shape. A skier renting pays 1 per day; buying costs B and ends it. Strip the dressing off all three and there's one object: a recurring cost of 1 per "day," a lump-sum cost B that ends the recurring cost forever, and a horizon — the number of days — that I don't get to see when I have to act. Let me just call it rent-or-buy and work in those units: rent = 1/day, buy = B, scaled so they're comparable.

What would I do if I could see the future? If I'll ski x days, the clairvoyant pays min(x, B): rent the whole time if x < B, buy on the first day if x ≥ B. That's the bar. C_OPT(x) = min(x, B). Everything I do on-line gets measured against this — not against some average over a presumed distribution of x, but against this same OPT on whatever x actually happens, in the worst case. That's the competitive-ratio lens Sleator and Tarjan set up for list update and paging: an algorithm is c-competitive if its cost is at most c·C_OPT plus a constant, for every input. They showed move-to-front stays within a factor of 2 of the best off-line list strategy on any sequence. I want the analogous constant here, and I want it to be the smallest possible.

So what are my actual choices on-line? At any moment I've been renting; I can keep renting or buy now. A pure strategy is just: pick a day a, rent through day a, buy on day a+1. Call it A_a. Its cost on horizon x is x if x ≤ a (the season ended before I bought), and a + B if x > a (I rented a days, then bought). Let me find the best a.

The adversary gets to pick x after seeing my a (I'm deterministic, so it can simulate me and know a exactly). It will pick the x that maximizes my ratio. If x ≤ a, I match OPT — ratio close to 1 — so the adversary won't do that. It'll push x just past a: x = a+1. Then I pay a + B, and OPT pays min(a+1, B). My ratio is (a+B)/min(a+1, B). To make this as small as I can, I want the denominator big, so I want a+1 ≥ B, i.e. a ≥ B−1; and I want the numerator small, so a as small as possible. Both pull me to a = B−1. Then I pay (B−1) + B = 2B−1, OPT pays B, ratio 2 − 1/B. This is the break-even rule: rent until the rent I've spent equals the purchase price, then buy. Spend B−1 on rent, and only then commit to the B.

Is 2 − 1/B the best any deterministic rule can do? Take any A_a and let the adversary stop at x = a+1, the first horizon on which I actually buy. The ratio is (a+B)/min(a+1, B). If a < B−1, that is (a+B)/(a+1), and it decreases as a moves toward B−1. If a = B−1, it is (2B−1)/B. If a ≥ B, OPT is already B and the ratio is at least (a+B)/B ≥ 2. So the minimum worst case is exactly at a = B−1, with ratio 2 − 1/B. Deterministically I'm stuck near 2. And I can feel exactly why: I commit to a single buy-day, the adversary knows which one, and it ends the season the instant after I buy. My determinism is the leak. The adversary plays against the one day I'll buy.

So plug the leak: don't buy on a fixed day. Randomize the buy-day. Now the adversary still has to commit to x in advance — it's oblivious, it sees my code and my distribution but not my coin flips — so it can no longer aim at "the day after I buy," because it doesn't know which day that is. If I mix two deterministic cutoffs, then against any fixed x my expected cost is a genuine average of two deterministic costs, and the adversary can't catch both cutoffs with one x. Let me check this even helps: with unscaled costs buy = 300 and rent = 20, mixing A_10 and A_15 half-and-half gives a worst-horizon expected cost (½)(500)+(½)(600) = 550 against OPT 300 — ratio 550/300 ≈ 1.83, already under 2. Good, randomization bites. The question is which distribution over buy-days is best, and how low I can push the ratio.

Let me set it up properly, and I need to be precise about the indexing or the constants will be off. Let π_i be the probability that I buy on day i, for i = 1, …, B. Buying on day i means I have rented i−1 days and, if the season reaches day i, I pay B at the start of that day. Buying after day B is wasteful: once I have already rented B days, OPT's long-horizon cost is B and delaying the purchase only adds rent before the same lump sum. Now fix a horizon k and compute my expected cost. A day-i buy costs (i−1)+B if i ≤ k, because the season reaches the buy day; it costs k if i > k, because the season ends before I buy. So

E[C(k)] = Σ_{i≤k} π_i (B+i−1) + (Σ_{i>k} π_i) · k.

What's OPT(k)? For k ≤ B, OPT just rents: OPT(k) = k. I want my expected cost to be at most some factor ρ times OPT for every k, and I want ρ as small as possible. If one horizon has a larger ratio than the others, the adversary plays that one; if some horizons are slack while another is tight, I have probably moved too much probability mass toward the slack side. So I try to flatten the ratio across all k. Set, for each k ≤ B,

E[C(k)] = ρk.

That's one equation per horizon — a whole system that pins down π. Let me extract the recurrence by looking at consecutive increments. Going from k to k+1, the term for i = k+1 flips from "season ended before I bought" to "the season reaches my buy day": it contributed π_{k+1}·k before and contributes π_{k+1}·(B+k) now, a change of Bπ_{k+1}. Every term with i > k+1 remains in the rent-only bucket and its multiplier rises from k to k+1, adding Σ_{i>k+1} π_i. So

E[C(k+1)] − E[C(k)] = Bπ_{k+1} + Σ_{i>k+1} π_i.

The right-hand side of the flattened constraint rises by ρ each time. Let T_k = Σ_{i>k} π_i, the tail mass above k. Then the increment equation is

Bπ_{k+1} + T_{k+1} = ρ.

The previous increment is Bπ_k + T_k = ρ, and T_k = π_{k+1}+T_{k+1}. Subtract the two equal increment equations:

Bπ_{k+1} + T_{k+1} = Bπ_k + π_{k+1} + T_{k+1}.

So

π_{k+1}(B−1) = Bπ_k,

and therefore

π_{k+1} = π_k · B/(B−1).

So the probabilities form a geometric sequence with ratio r = B/(B−1) = 1/(1 − 1/B) > 1 — the buy-day probabilities *grow* as the day gets later, increasing toward the break-even deadline. That sits right: the longer I've already been renting without the season ending, the more I should be willing to commit and buy. The mass piles up near day B.

Pin the first term from the boundary. Starting from horizon 0 to horizon 1 gives the same increment equation: the day-1 buy mass pays B, and all later buy-days just rent the one day, so ρ = Bπ_1 + (1−π_1) = 1 + (B−1)π_1. If I call the normalizing first mass α = π_1, then

π_i = α r^{i−1},  r = B/(B−1).

Now α isn't free — the π_i must sum to 1. Since r−1 = 1/(B−1),

Σ_{i=1}^{B} π_i = α Σ_{i=1}^{B} r^{i−1}
                = α (r^B−1)/(r−1)
                = α (B−1)(r^B−1).

Set that to 1:

α = 1 / ((B−1)(r^B−1)).

The competitive factor comes from the boundary ρ = 1 + (B−1)α. Substitute the normalized α:

ρ = 1 + 1/(r^B−1) = r^B/(r^B−1).

What is r^B? It's (B/(B−1))^B = (1 − 1/B)^{−B}. So

ρ = r^B/(r^B−1) = 1/(1 − (1−1/B)^B),

and as B grows, (1−1/B)^B → 1/e, hence ρ → e/(e−1) ≈ 1.58.

The flattening forced a geometric distribution on the buy-day, and the normalization forced the ratio. I didn't pick 1.58 — it fell out of "make every horizon equally bad and let the probabilities sum to one." Let me sanity-check the small cases against the limit: B = 5 gives ρ ≈ 1.487, B = 10 gives ≈ 1.535, B = 100 gives ≈ 1.577, all sliding up toward 1.582 from below as B → ∞. And it's well under the deterministic 2 − 1/B for every B ≥ 2. The drop from 2 to e/(e−1) is the whole payoff of hiding the buy-day behind a coin.

Before I trust it, let me make sure horizons k > B can't bite. For k > B, OPT(k) = B because the clairvoyant buys on day one. My expected cost is now independent of k: every buy-day i ≤ B is reached, so E[C] = Σ_i π_i(B+i−1). But that is exactly E[C(B)], since horizon B also reaches every buy-day in the support. The flattened constraint at k = B already made E[C(B)] = ρB. So no horizon beyond B is worse than the ones I equalized. Good — and this also confirms why the support must stop at buy day B, not at "after renting B days." One extra delayed day would leave the k = B equation looking fine while making every longer horizon pay an unnecessary extra rent day.

Now, is this actually optimal, or did I just find a good distribution? I need a matching lower bound: no algorithm, however clever, beats e/(e−1) against the oblivious adversary.

Yao's principle is the right tool because it lets me lower-bound randomized algorithms by choosing a fixed distribution over horizons and asking how the best deterministic cutoff performs on average. So I want a horizon distribution X that makes every deterministic cutoff equally unattractive. A deterministic A_a rents a days and buys on day a+1, so for any random X,

E[c(A_a,X)] = Σ_{t=1}^{a} Pr[X ≥ t] + B Pr[X > a]
             = Σ_{t=1}^{a} Pr[X ≥ t] + B Pr[X ≥ a+1].

To make A_a and A_{a−1} tie, subtract the two expected costs:

Pr[X ≥ a] + BPr[X ≥ a+1] = BPr[X ≥ a].

So the tail must obey

Pr[X ≥ a+1] = (1 − 1/B)Pr[X ≥ a].

With Pr[X ≥ 1] = 1, this gives

Pr[X ≥ t] = (1 − 1/B)^{t−1}.

Under this X, every deterministic cutoff has the same expected cost:

E[c(A_a,X)] = Σ_{t=1}^{a} (1−1/B)^{t−1} + B(1−1/B)^a
             = B(1 − (1−1/B)^a) + B(1−1/B)^a
             = B.

The clairvoyant pays min(X,B), whose expectation is the sum of the first B tail probabilities:

E[c(OPT,X)] = Σ_{t=1}^{B} Pr[X ≥ t]
             = B(1 − (1−1/B)^B).

So

min_a E[c(a,X)] / E[c(OPT,X)] = B / [B(1 − (1−1/B)^B)] = 1/(1 − (1−1/B)^B) → 1/(1 − 1/e) = e/(e−1).

By Yao, every randomized algorithm has worst-case ratio at least this. The upper bound used r = 1/(1−1/B) to spread buy-days; the lower bound uses the reciprocal tail factor 1−1/B to spread horizons. The two sides meet at exactly 1/(1 − (1−1/B)^B), and the limit is e/(e−1).

Now the continuous version, the spin-block problem, because the lock-waiting time isn't an integer count of days — it's a real number, spin cost ∝ time, block cost B. Scale B = 1 so the break-even point is time 1. Same logic, but the distribution over the buy-time z becomes a density on [0,1]. Let p(z) be that density. If the lock is held until time τ ≤ 1, OPT pays τ, and my expected cost is

E[C(τ)] = ∫_0^τ p(z)(z+1) dz + τ∫_τ^1 p(z) dz.

Flattening means E[C(τ)] = ρτ for every τ in [0,1]. Differentiate once:

ρ = p(τ)(τ+1) + ∫_τ^1 p(z) dz − τp(τ)
  = p(τ) + ∫_τ^1 p(z) dz.

Differentiate again:

0 = p′(τ) − p(τ),

so p′ = p and p(z) = Ce^z. The discrete recurrence π_{k+1}/π_k = 1/(1−1/B) was the finite-difference shadow of exactly this exponential growth. Normalize over [0,1]:

∫_0^1 Ce^z dz = C(e−1) = 1,

so C = 1/(e−1) and

p(z) = e^z/(e−1) on [0,1].

The factor is the boundary value of the derivative equation at τ = 1: the tail integral vanishes, so ρ = p(1) = e/(e−1). Same constant, now exact rather than a limit. The truncation at z = 1 matters for the same reason it did discretely: any density mass beyond the break-even time is pure waste, so the support stops at 1.

Step back and I can name the whole causal chain. The pain is a recurring-vs-lump-sum decision with an adversarial, hidden horizon. Measured against the clairvoyant min(x, B), the deterministic freedom is the buy-day, the best cutoff is break-even (rent B−1 days, then buy on day B), and that's stuck near 2 because the oblivious adversary can aim at the one day I'll buy. Randomizing the buy-day removes that aim. Demanding that the expected ratio be flat across every horizon forces the buy-day probabilities to be geometric with ratio 1/(1−1/B), growing toward the deadline; normalization gives π_i = αr^{i−1} with α = 1/((B−1)(r^B−1)), and the boundary equation gives competitive factor ρ = r^B/(r^B−1) = 1/(1 − (1−1/B)^B) → e/(e−1) ≈ 1.58. The matching lower bound is the reciprocal geometry seen from the adversary's side via Yao, with horizon tail (1−1/B)^{t−1} making every deterministic cutoff equally bad. In continuous time the geometric becomes the exponential density e^z/(e−1) on [0,1], with the constant exact. Now let me write it as code: OPT, the deterministic buy days, the geometric buy-day distribution, the worst-horizon ratio, and the continuous sampler.

```python
import math, random

def opt_cost(x, B):
    # clairvoyant: rent everything if the horizon is short, else buy day one
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
