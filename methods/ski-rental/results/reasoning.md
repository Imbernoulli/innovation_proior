OK, the thing that actually hurts is the hidden stopping time. I have a block of memory shared between two caches. One processor keeps writing to it, and every write has to be broadcast on the bus so the other cache stays coherent — that's one bus cycle per write, a steady drip of cost. I could stop the drip by invalidating the block in the other cache, but invalidation isn't free: when the other processor eventually reads that block again, it has to be re-fetched, and that re-fetch costs me the whole block transfer, call it p cycles. So I'm paying 1 per write to keep sharing, or I pay a one-time p to stop. If only I knew how many more writes were coming before the other cache wants the block back, I'd know exactly what to do. I don't.

And it's not just caching. A process spinning on a lock pays per unit of time it spins; or it can block and eat a fixed context-switch cost C. Same shape. A skier renting pays 1 per day; buying costs B and ends it. Strip the dressing off all three and there's one object: a recurring cost of 1 per "day," a lump-sum cost B that ends the recurring cost forever, and a horizon — the number of days — that I don't get to see when I have to act. Let me just call it rent-or-buy and work in those units: rent = 1/day, buy = B, scaled so they're comparable.

What would I do if I could see the future? If I'll ski x days, the clairvoyant pays min(x, B): rent the whole time if x < B, buy on the first day if x ≥ B. That's the bar. C_OPT(x) = min(x, B). Everything I do on-line gets measured against this — not against some average over a presumed distribution of x, but against this same OPT on whatever x actually happens, in the worst case. That's the competitive-ratio lens Sleator and Tarjan set up for list update and paging: an algorithm is c-competitive if its cost is at most c·C_OPT plus a constant, for every input. They showed move-to-front stays within a factor of 2 of the best off-line list strategy on any sequence. I want the analogous constant here, and I want it to be the smallest possible.

So what are my actual choices on-line? At any moment I've been renting; I can keep renting or buy now. A pure strategy is just: pick a day a, rent through day a, buy on day a+1. Call it A_a. Its cost on horizon x is x if x ≤ a (the season ended before I bought), and a + B if x > a (I rented a days, then bought). Let me find the best a.

The adversary gets to pick x after seeing my a (I'm deterministic, so it can simulate me and know a exactly). It will pick the x that maximizes my ratio. If x ≤ a, I match OPT — ratio close to 1 — so the adversary won't do that. It'll push x just past a: x = a+1. Then I pay a + B, and OPT pays min(a+1, B). My ratio is (a+B)/min(a+1, B). To make this as small as I can, I want the denominator big, so I want a+1 ≥ B, i.e. a ≥ B−1; and I want the numerator small, so a as small as possible. Both pull me to a = B−1. Then I pay (B−1) + B = 2B−1, OPT pays B, ratio 2 − 1/B. This is the break-even rule: rent until the rent I've spent equals the purchase price, then buy. Spend B−1 on rent, and only then commit to the B.

Is 2 − 1/B the best any deterministic rule can do? Take any A_a and let the adversary stop at x = a+1, the first horizon on which I actually buy. The ratio is (a+B)/min(a+1, B). If a < B−1, that is (a+B)/(a+1), and it decreases as a moves toward B−1. If a = B−1, it is (2B−1)/B. If a ≥ B, OPT is already B and the ratio is at least (a+B)/B ≥ 2. So the minimum worst case is exactly at a = B−1, with ratio 2 − 1/B. Deterministically I'm stuck near 2. And I can feel exactly why: I commit to a single buy-day, the adversary knows which one, and it ends the season the instant after I buy. My determinism is the leak. The adversary plays against the one day I'll buy.

So plug the leak: don't buy on a fixed day. Randomize the buy-day. Now the adversary still has to commit to x in advance — it's oblivious, it sees my code and my distribution but not my coin flips — so it can no longer aim at "the day after I buy," because it doesn't know which day that is. If I mix two deterministic cutoffs, then against any fixed x my expected cost is a genuine average of two deterministic costs, and the adversary can't catch both cutoffs with one x. With unscaled costs buy = 300 and rent = 20, mixing A_10 and A_15 half-and-half gives a worst-horizon expected cost (½)(500)+(½)(600) = 550 against OPT 300 — ratio 550/300 ≈ 1.83, already under 2. So randomization does bite; the question is which distribution over buy-days is best, and how low I can push the ratio.

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

The flattening forced a geometric distribution on the buy-day, and the normalization forced the ratio. I didn't pick 1.58 — it fell out of "make every horizon equally bad and let the probabilities sum to one." The small cases track the limit from below: B = 5 gives ρ ≈ 1.487, B = 10 gives ≈ 1.535, B = 100 gives ≈ 1.577, sliding up toward 1.582 as B → ∞, and well under the deterministic 2 − 1/B for every B ≥ 2. The drop from 2 to e/(e−1) is the whole payoff of hiding the buy-day behind a coin.

Horizons k > B need checking too, since I only flattened k ≤ B. There OPT(k) = B because the clairvoyant buys on day one, and my expected cost is now independent of k: every buy-day i ≤ B is reached, so E[C] = Σ_i π_i(B+i−1) — which is exactly E[C(B)], already pinned to ρB by the k = B constraint. So no horizon beyond B is worse than the ones already equalized, and this is also why the support must stop at buy day B rather than "after renting B days": one extra delayed day would leave the k = B equation looking fine while making every longer horizon pay an unnecessary extra rent day.

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

Time to make it concrete: OPT, the deterministic break-even baseline, the geometric buy-day distribution with its sampler, the Yao adversary's tail probability, and the continuous density with its inverse-CDF sampler, so I can sweep B and watch both ratios converge — 2 − 1/B for the deterministic cutoff, and 1/(1 − (1−1/B)^B) sliding up to e/(e−1) for the randomized one. The full module is in the answer.
