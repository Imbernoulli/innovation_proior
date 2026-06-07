# Context: rent-or-buy under an unknown, adversarial horizon

## Research question

A decision-maker faces a stream of identical opportunities and at each one must pay a small recurring cost or, once, a large lump sum that abolishes the recurring cost forever. Renting a pair of skis costs 1 per day; buying costs B and ends all future rental. A snoopy cache broadcasting a shared block pays 1 per write to keep it coherent, or pays a penalty B to invalidate it once and stop the broadcasts. A process waiting on a lock spins at a cost proportional to the time it waits, or blocks at a fixed context-switch cost B. In every case the same question: how long do you keep paying the recurring cost before you pay the lump sum?

The difficulty is that the right answer depends on the horizon — how many more days you will ski, how many more writes will hit the shared block, how long until the lock frees — and that horizon is unknown when you must act, and may be chosen by an adversary to punish whatever you decide. The goal is an on-line rule whose cost, on every input, is within the smallest possible constant factor of the cost a clairvoyant who knew the horizon would have paid. The question is what that smallest factor is, and what rule achieves it.

## Background

The yardstick is competitive analysis, introduced by Sleator and Tarjan (1985) in their study of list-update and paging rules. They measured an on-line algorithm not against an average-case input distribution but against the optimum off-line cost on the *same* worst-case sequence: an algorithm A is c-competitive if there is a constant δ with C_A(σ) ≤ c · C_OPT(σ) + δ for every input σ, where C_OPT is the cost of the best clairvoyant strategy. Their move-to-front result — total cost within a factor of 2 of the optimum off-line list-maintenance strategy on any access sequence — established the worst-case-ratio-against-OPT framing and showed that a simple on-line rule can be provably within a constant of optimum. The constant 2 recurs throughout this line of work.

The rent-or-buy structure is the simplest non-trivial instance. Scale so that one day of renting costs 1 and buying costs B (an integer). The clairvoyant who knows the horizon x pays C_OPT(x) = min(x, B): rent the whole time if you will ski fewer than B days, buy on day one otherwise. The whole problem lives in the gap between this and what an on-line player, blind to x, can guarantee.

Two notions of adversary matter, and the gap between them is the central phenomenon. A *strong* (adaptive) adversary picks each request after seeing the algorithm's previous moves, including the outcomes of any coin flips. A *weak* (oblivious) adversary must commit to the whole input sequence in advance, knowing the algorithm's code but not the realizations of its randomness. For a deterministic algorithm the two are the same — an oblivious adversary can simulate the algorithm on paper and predict every move — so randomization can only help against the weak adversary. Whether some problem has a strictly smaller competitive factor against a weak adversary than against a strong one was, at the time, a live and surprising question; a few dramatic separations were known (an n-state metrical task system improving from Θ(n) to Θ(log n); uniform-distance paging improving from k to the k-th harmonic number Hₖ), but no such separation was known for a server problem on a non-uniform graph, of which rent-or-buy is the smallest case.

The tool for lower-bounding randomized algorithms is Yao's minimax principle. For a finite family of deterministic algorithms A and a finite family of instances X, and any distribution over instances, the worst-case expected cost of any randomized algorithm is at least the expected cost of the best deterministic algorithm under that instance distribution: maxₓ E[c(A,x)] ≥ min_a E[c(a,X)]. The left side is what we want to bound — the worst-case expected cost of the best randomized algorithm. The right side replaces the hard object (a randomized algorithm) with an easy one (the best deterministic algorithm against a fixed, cleverly chosen input distribution). Choosing the input distribution well turns a randomized lower bound into an averaging argument over deterministic strategies.

A motivating empirical observation sharpens the question. Traces of programs on snoopy-caching multiprocessors (Eggers and Katz, 1989) show that write-run lengths are wildly different across programs but very stable within a program: either a block is actively shared (very short runs) or it is essentially private (very long runs), with little in between, and the inter-program variability is high while the intra-program variability is low. This suggests that an algorithm which adapts to the observed statistics could converge to near-optimal behavior on any one program even though no single fixed cutoff is good across all programs.

## Baselines

**Best fixed-cutoff deterministic rule.** The natural deterministic strategies are a one-parameter family: A_a rents for a days and buys on day a+1. Its cost on horizon x is x if x ≤ a, and a + B otherwise. The worst horizon for A_a is x = a + 1: the player rents a days, buys, then never needs the skis again, paying a + B while OPT pays min(a+1, B). If a < B − 1, this ratio is (a+B)/(a+1), which decreases as a approaches B−1. If a ≥ B, the horizon x = a+1 gives ratio at least (a+B)/B ≥ 2. The best cutoff is therefore a = B − 1 — rent until cumulative rent is just below the purchase price — yielding cost 2B − 1 against OPT = B, a ratio of 2 − 1/B. The gap it leaves open: an oblivious adversary, knowing the deterministic player will buy exactly on day B, simply ends the horizon on day B, forcing the full 2 − 1/B every time. Determinism is the weakness the adversary exploits.

**Always-buy / always-rent.** Buying on day one is 1-competitive when the horizon is long but unbounded-competitive when the horizon is 1 (you pay B, OPT pays 1). Renting forever is 1-competitive when the horizon is short but unbounded when it is long (you pay x, OPT pays B). Neither bounds the ratio; both are dominated by the break-even rule, which is why the cutoff is the only sensible degree of freedom.

**Naive mixtures of two cutoffs.** Splitting probability between two cutoffs, e.g. choosing A_{a₁} with probability ½ and A_{a₂} with probability ½, already beats 2: against the worst horizon the expected cost is a strict average of two deterministic costs and the adversary can no longer aim at a single known cutoff. This shows randomization helps but leaves open which distribution over cutoffs is best and how far below 2 one can push.

## Evaluation settings

The natural yardstick is the worst-case expected competitive ratio: for a candidate (possibly randomized) rule, maxₓ E[C_A(x)] / C_OPT(x) over all horizons x, with C_OPT(x) = min(x, B). The horizon ranges over the integers (days, writes) in the discrete model and over the non-negative reals (waiting time) in the continuous spin-block model. The instance family for a Yao-style lower bound is the same set of horizons, equipped with an adversary-chosen distribution. The recurring/lump-sum ratio B (= p, the block size, in the cache model; = C, the context-switch cost, in the spin model) is the single parameter, and the regime of interest is B → ∞, where the constants converge. A simulation would sweep B, compute the exact worst-horizon expected ratio of a candidate distribution, and Monte-Carlo-sample buy days against each horizon to confirm the closed form.

## Code framework

```python
import math, random

def opt_cost(x, B):
    # clairvoyant: rent all if horizon short, buy day one if long
    return min(x, B)

def det_cost(buy_day, x, B):
    # buy_day is 1-indexed: rent buy_day-1 days, then buy if the horizon reaches it
    return x if x < buy_day else (buy_day - 1) + B

def best_det_ratio(B):
    # worst-case ratio of the break-even buy day B
    # TODO: return the deterministic competitive factor
    pass

def buy_day_distribution(B):
    # the on-line player's randomization over buy days {1,...,B}:
    # TODO: choose the probability of each buy day so the
    #       per-horizon competitive ratio is as small (and as flat) as possible
    pass

def expected_ratio_exact(B):
    # worst over horizons x of E_i[det_cost(i, x, B)] / opt_cost(x, B)
    # TODO: evaluate the candidate distribution
    pass

def sample_buy_day(pi, rng):
    # TODO: sample one buy day from the distribution before play begins
    pass

def adversary_tail(t, B):
    # TODO: Yao lower-bound tail probability Pr[X >= t]
    pass

def continuous_density(z):
    # TODO: density for the continuous spin/block buy time after scaling B to 1
    pass

def sample_buy_time_continuous(rng):
    # TODO: inverse-CDF sampler for the continuous buy time
    pass

def continuous_ratio(z, d):
    # TODO: cost ratio for buy time z and continuous horizon d
    pass

if __name__ == "__main__":
    for B in [5, 10, 100, 1000]:
        print(B, best_det_ratio(B), expected_ratio_exact(B))
```
