The selection rule is the whole point, but it sits inside a fixed loop, and with the default fill that
loop is the floor — uniform-random arm selection, which spends every round exploring and never
exploits. So the pain to start from is just selecting arms at all under the harness's one constraint
that matters here: a single rule, computed from running per-arm statistics, has to serve a stationary
10-arm Bernoulli MAB, a contextual bandit, and a non-stationary bandit, with `T = 10000` rounds on each
and normalized regret as the score. I want the simplest principled rule that is provably better than
random, and I want to understand precisely what it will and won't handle, because that is what the next
rung has to fix.

Let me write down what I am actually optimizing so I don't fool myself. Regret is
`R_T = mu* T - sum_a mu_a E[N_a(T)]`, and since the pull counts sum to `T`, this collapses to
`R_T = sum_{a: mu_a < mu*} Delta_a E[N_a(T)]` with `Delta_a = mu* - mu_a` the gap of arm `a`. The means
and gaps are fixed by the environment; the only thing the policy controls is `E[N_a(T)]`, the expected
number of pulls of each suboptimal arm. The harness then divides by `T`, so my score on each setting is
literally a gap-weighted average of how often I pull bad arms. The entire game is: pull the bad arms as
few times as I can get away with, while still pulling each one enough to be sure it really is bad.

I can make this concrete for the one world whose arms I am handed explicitly. On the stochastic MAB the
means are `[0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]`, so `mu* = 0.80` and the nine
gaps are `[0.70, 0.60, 0.50, 0.45, 0.40, 0.30, 0.25, 0.20, 0.10]`, summing to `3.5`. Two things jump
out. First, the reciprocals `1/Delta_a` sum to about `32.15`, and that sum is dominated by the near-best
arms: the arm at `0.70` alone contributes `10` and the arm at `0.60` contributes `5`, while the arm at
`0.10` contributes only `1.43`. So whatever index I build, the regret will be paid overwhelmingly on the
*small-gap* arms clustered just below the optimum — those are the ones that are genuinely hard to
distinguish from the best and cost the most per unfair pull. Second, the uniform-random floor's cost is
now computable: a random pull lands on each arm with probability `1/10`, so its expected instantaneous
regret is the mean gap `3.5/10 = 0.35`, and over all `T = 10000` rounds that is `3500` cumulative,
normalized `0.35`. That is the number I have to beat, and it is enormous — the floor throws away a third
of the achievable reward.

The obvious thing — estimate each mean by its empirical average `mu_hat_a = rewards_a / counts_a` and
always play the largest — I can watch die. Suppose the truly-best arm, on its first couple of pulls,
returns unlucky low rewards; Bernoulli `[0,1]` rewards are noisy, this happens routinely. Its empirical
mean drops below some mediocre arm's, the greedy rule stops choosing it, and here is the trap: because
greedy never pulls it again, its estimate never gets another sample and stays buried forever. The agent
locks onto a worse arm and pays its gap on every remaining round — regret `Theta(T)`, linear, the worst
possible, no better than the uniform-random floor in the limit and arguably worse because it commits.
The lesson is sharp: an arm I have sampled few times, I genuinely don't know about, and I cannot let a
noisy estimate permanently exile it.

How much willingness to revisit, though? The crude fix is to flip a coin — with probability `1-eps`
play the empirical best, with probability `eps` play a uniform random arm. That breaks the lock-in, but
price it exactly. A constant `eps` means a constant fraction of *all* rounds are uniformly random
forever, and each such random round pays the mean gap `0.35` in expectation, so the exploration alone
floors my cumulative regret at `eps * T * 0.35`; at `eps = 0.1` that is `0.1 * 10000 * 0.35 = 350`, a
normalized `0.035` I can never get below, and it grows linearly with `T`. To make regret sublinear I
would have to shrink `eps` as I learn, but tuning the decay rate correctly to guarantee a logarithmic
bound needs a lower bound on the smallest gap `Delta` — here `0.10`, but the harness never tells me that
and it differs across the three settings. So undirected exploration — exploring by coin flip, uniformly,
blind to which arms are actually uncertain — wastes pulls on arms I am already sure are bad. Notice the
waste is worst exactly where I said the regret lives: the coin spends as many exploratory pulls on the
`0.10`-mean arm (which I can rule out after a handful of samples) as on the `0.70`-mean arm (which is
genuinely close to optimal). I want exploration that is *targeted*: spend exploratory pulls on the arms
whose value I am genuinely unsure about, and stop spending them once I am sure. The uncertainty itself
should decide where to explore.

So I need, per arm, not just a point estimate `mu_hat_a` but a sense of how *trustworthy* that estimate
is. An arm pulled a thousand times — I know its mean well. An arm pulled twice — I barely know it; its
true mean could plausibly be much higher than the two samples suggest. The move is to compare arms not
by their empirical means but by an *optimistic* estimate: for each arm, the highest value its true mean
could plausibly take given the data, and play the largest of those. Picture what this does. A genuinely
good, well-sampled arm has a tight estimate near its true high mean, so its optimistic value is high — I
exploit it. A genuinely bad, well-sampled arm has a tight estimate near its true low mean, so its
optimistic value is low — I correctly leave it alone. The interesting case is an under-sampled arm: its
estimate is loose, so its optimistic value is inflated well above its empirical mean, which pulls me
toward sampling it. And if that optimism was misplaced — the arm really is bad — pulling it tightens its
estimate and the inflated optimistic value collapses toward the true low mean, so I quickly stop. But if
the optimism was warranted — the arm really is good and I had under-sampled it — I keep pulling and reap
it. Either way the policy self-corrects: a wrongly-high optimistic value cannot survive being acted on.
That is the property `eps`-greedy lacked — exploration directed by uncertainty, self-extinguishing when
the uncertainty resolves the wrong way. Be optimistic in the face of uncertainty, then act greedily on
the optimism.

Put this on the exact failure that killed greedy. Say each arm has been played once and the best arm
(mean `0.80`, arm 9) drew a `0` while the second-best (mean `0.70`, arm 8) drew a `1`. Greedy has
`mu_hat_9 = 0`, `mu_hat_8 = 1` and picks arm 8 forever. Under the optimistic rule both arms have `N = 1`
and carry the *same* bonus, call it `b(1)`; arm 8 wins this round at `1 + b(1)`, but arm 9 sits second at
`0 + b(1)` and its index does not decay while `N_9` stays at `1`. As I keep pulling arm 8 its mean
regresses from the fluke `1` toward `0.70` and its bonus shrinks, while arm 9's `b(1)` if anything rises
with the round count — so the curves cross, arm 9 gets re-pulled, and the permanent exile greedy suffered
cannot happen: any small-count arm keeps an inflated index that forces a revisit.

Now I have to make "the highest plausible value of `mu_a`" precise, and that is a confidence interval.
I need: how far above the empirical mean `mu_hat_a` could the true mean `mu_a` be, with overwhelming
probability, given `N_a` samples? Concentration answers exactly this. The Chernoff–Hoeffding bound says
that for `s` samples in `[0,1]` with mean `mu`, the empirical mean concentrates with sub-Gaussian tails:
`P(mu_hat >= mu + a) <= exp(-2 s a^2)` and symmetrically below. So to make a one-sided confidence
statement that fails with probability at most `delta`, set `exp(-2 s a^2) = delta`, i.e.
`a = sqrt(ln(1/delta) / (2 s))`. That is the half-width. The upper confidence bound on `mu_a` is then
`mu_hat_a + a`, and I play the arm maximizing it. The radius already has the two behaviours I wanted: it
shrinks like `1/sqrt(s)` as I pull an arm more (its optimistic value descends toward truth — exploit
good arms, abandon bad ones), and if I let `delta` grow with time, it creeps back up for arms I have
stopped pulling, so no arm is dismissed forever. This is exactly the `b(N)` I invoked in the trace, now
pinned down: `b(1) = sqrt(ln(1/delta)/2)`, large, which is why a once-pulled arm cannot be exiled.

What should `delta` be? Here is the tension. I want `delta` tiny, so the confidence intervals
essentially never fail — a failure means my optimism *under*-estimated the best arm, which is how I would
wrongly abandon it. But I apply this interval at every round, for every arm, for every possible sample
count, so when I bound the total probability of any failure over the whole run with a union bound, I am
summing a lot of `delta`s. A constant `delta` makes that sum diverge, so `delta` must shrink with time
fast enough that the union-bounded total stays finite. Tie `delta` to the round index: at round `t`,
set `delta = t^{-4}`, so `ln(1/delta) = 4 ln t` and the radius becomes `sqrt(4 ln t / (2 s)) =
sqrt(2 ln t / s)`. Why the `4`? It has to come out of making the regret proof's union bound converge,
and it does: with `delta = t^{-4}` each Hoeffding failure event has probability `t^{-4}`, the union
bound leaves `~t^2` events per round (a round index times a sample count, each up to `t`), so the
per-round contribution is `~t^2 * t^{-4} = t^{-2}`, whose sum over `t` converges (to `pi^2/6`); a milder
`delta = t^{-2}` would leave `~1` per round and the bound would collapse linearly. So the exponent `4` is
the smallest clean choice that makes the post-union series converge, and it produces the canonical
confidence radius `sqrt(2 ln t / N_a)`.

The policy, then: first play each arm once so every `N_a >= 1` and the radius is well-defined, then at
each round play the arm `a` maximizing `mu_hat_a + sqrt(2 ln t / N_a)`. Empirical mean plus an
uncertainty bonus; argmax; that is it. The round-robin prologue costs almost nothing — on the stochastic
world it pulls each of the nine suboptimal arms once, so its whole regret contribution is `sum Delta_a =
3.5` cumulative, a rounding error against `T = 10000`. The steady-state rule is cheap too: each round
just recomputes `K` indices from running sums, `O(K)`, and it assumes nothing about the reward
distributions beyond support in `[0,1]`. Running the counting argument (bound the rounds a suboptimal
arm's index can beat the optimum's, union-bound over all sample counts, split each such round into "the
optimal arm was under-estimated," "the bad arm was over-estimated," both `<= t^{-4}` by Hoeffding, or
"the bad arm's interval was still too wide," which becomes impossible once `N_a >= 8 ln T / Delta_a^2`)
yields `E[N_a(T)] <= 8 ln T / Delta_a^2 + 1 + pi^2/3`, hence `R_T <= 8 sum_a (ln T / Delta_a) +
(1+pi^2/3) sum_a Delta_a`. Logarithmic, finite-time, distribution-free over `[0,1]` — exactly the
finite-horizon guarantee the asymptotic Lai–Robbins index policies left open.

Put the stochastic MAB's numbers through this bound, to calibrate how loose it is. The leading term is `8 * ln(10000) * sum_a (1/Delta_a) = 8 * 9.21 * 32.15 ≈
2369`, and the lower-order term is `(1 + pi^2/3) * 3.5 ≈ 15`, so the bound predicts cumulative regret
`≈ 2384`, normalized `≈ 0.24`. That is better than the random floor's `0.35` but not by an impressive
margin — and it *should* be loose, because the counting argument spends the constant `8` and, via
Pinsker, `8/Delta_a^2 = 16 * (1/(2 Delta_a^2))` is sixteen times the information-theoretic `1/(2
Delta_a^2)` that a mean-`Delta_a` arm ideally costs. The gap-free reading tells the more optimistic
story: splitting arms at a gap threshold `gamma` (small-gap arms cost at most `T*gamma` total, large-gap
arms use the log bound, balanced at `gamma = sqrt(8 K ln T / T)`) gives `O(sqrt(K T ln T))`, and here
`sqrt(10 * 10000 * 9.21) ≈ 960` cumulative. So somewhere between the loose `2384` gap-dependent bound and
this `960` gap-free bound is where the guarantee lives, both worst-case envelopes — the realized number
is typically well under either, since Hoeffding's `8` is pessimistic and the arms are not adversarially
placed. I expect a few percent. Either way `R_T/T -> 0` on the stationary world, the qualitative jump
from the random floor I am replacing.

Now I have to be careful about *this* harness rather than the textbook, because the rule is graded on
three different worlds and the contract only gives me `select_arm(t, context)` and `update(arm, reward,
context)`. The literal edit I will land is the plain UCB1 index above, applied identically on all three
settings — `t < K` round-robin first, then `argmax_a mu_hat_a + sqrt(2 ln(t+1) / N_a)` over the full
history. I deliberately keep the `+1` inside the log (`ln(t+1)`) so the first post-initialization rounds
have a finite, well-scaled bonus rather than `ln` of a small integer: at the first index round on the
K=5 settings that is `ln(6) = 1.79` instead of `ln(5) = 1.61`, a small shift, but it keeps the bonus
strictly positive and smooth right out of the round-robin. I considered specializing per regime — the
natural temptations are a sliding-window UCB for the non-stationary setting (recompute the index over
only the last `W` pulls so stale segments are forgotten) and some use of the context vector on the
contextual setting. On the non-stationary world a sliding window is the textbook move, and I even carry
the buffer machinery in `update` so the option is one call away; but when I actually reason about what it
costs, the window hurts the *stationary* and *contextual* worlds badly. Trace the mechanism: with a
window of size `W`, an arm's confidence radius is governed by its count *within the window*, not its
lifetime count. UCB1 rightly concentrates almost all pulls on the best arm, so on the stochastic MAB the
best arm fills most of the `W` window slots while each suboptimal arm holds only a handful; their
in-window counts never grow, so their radii stay permanently wide and never tighten, and worse, once a
confirmed-bad arm goes `W` rounds without a pull its window stats decay to zero and it registers as
"unpulled," forcing a fresh re-exploration. That happens roughly every `W` rounds, so over `T = 10000`
with `W = 800` there are `~12` forced re-exploration cycles, each re-paying a chunk of gap-weighted
regret on arms I had already ruled out — enough to turn the theoretical `~960` cumulative regret on the
stationary MAB into something closer to `~1450`. Since one rule is graded on all three and I cannot see
which world I am in, importing a sliding-window fallback would trade a large, certain loss on two
settings for a speculative gain on one. So the honest baseline is *plain* UCB1 everywhere: I leave the
circular-buffer running stats maintained in `update` (cheap, `O(1)`), but `select_arm` never consults
them — the index is always the full-history UCB1. On the contextual setting I make no use of the context
at all; `select_arm` ignores it and runs per-arm UCB1 on the marginal reward. This is knowingly
suboptimal there, but it is the clean, single-rule baseline, and naming exactly where it is blind is the
point of starting here.

So at step 1 the rule is settled and my edit replaces the uniform-random placeholder with full-history
UCB1: round-robin once, then optimistic-index argmax, with the running sums in `update` and the (dormant)
sliding-window buffer maintained but unused (the full module is in the answer).

Now reason about what this floor must do, because that is the entire point of running it. On the
**stochastic MAB** UCB1 is in its home regime: stationary Bernoulli arms, bounded rewards, the index
provably logarithmic. I expect it to do well — normalized regret on the order of a few percent, far
below both the `0.24` loose bound and the `0.35` random floor, with the `8/Delta_a^2` constant being a
factor off the tightest possible index, so "good but not the tightest." The regret it does pay will be
concentrated on the small-gap arms at `0.70`, `0.60`, `0.55`, exactly the `1/Delta_a`-heavy ones I
flagged, because those are the ones whose confidence intervals overlap the optimum longest. On the
**contextual** setting I expect it to be poor, and I can sharpen *why* from the environment's structure.
The reward of arm `a` is `x·theta_a + noise` with `x` drawn on the unit sphere, so the *marginal* mean
UCB1 estimates — averaging over the random `x` — is `E_x[x·theta_a] = theta_a · E_x[x] = 0`, because a
symmetric unit-sphere context has `E_x[x] = 0`. Every arm's marginal is the same (a small positive
number once the `[0,1]` clipping folds in the negative side), so the five per-arm averages UCB1 tracks
collapse onto each other and it has essentially no signal to separate the arms — it degenerates toward
random selection. Meanwhile the *real* per-round best arm is `argmax_a x·theta_a`, which swings with `x`
every round; there is no fixed best arm to converge to, and the per-round gaps are real but invisible to
a rule that only holds one scalar per arm. So I expect large regret here (high teens of a percent), among
the worst of any sensible rule, and structurally uncloseable without modeling `x`. On the
**non-stationary** setting I expect it to be the worst of the three worlds for UCB1: the index is built
on the *full* history, so after a changepoint the empirical mean of the formerly-best arm is anchored by
thousands of stale samples and the confidence radius is tiny, so it takes a long time to unlearn the old
best and discover the new one. To feel the timescale, take an illustrative case: if the old best had
accumulated `~1500` pulls at `mu_hat ≈ 0.8` and its true mean has just dropped, then after `k` fresh
contrary pulls its estimate is `1500 * 0.8 / (1500 + k)`, and driving that below even `0.5` needs `k >
900` — nine hundred pulls of a now-bad arm, each paying its gap, before its index even stops dominating.
Four changepoints means four such slow relearnings, each accruing gap-sized regret in the interim, so I
expect normalized regret clearly worse than its stationary number.

Whatever the precise split, the diagnosis is already pointed at the next rung. UCB1's weaknesses are not
a failure of the optimism principle; they are two specific mismatches. First, its confidence radius is
*range-only* — it depends on `N_a` and the `[0,1]` support but never on *where* `mu_hat_a` sits, so a
near-zero arm and a fair-coin arm get the same width at equal counts, which over-explores on the
stochastic world relative to the tighter geometry that the *shape* of a Bernoulli mean would license.
Second, it has *no mechanism* for either context or non-stationarity beyond the one full-history index.
So the next move is to keep the optimism but sharpen the confidence geometry — replace the symmetric
Hoeffding radius with one that self-tightens where the empirical mean sits near the boundaries and, in
particular, does something about stale history where the world has changed. That is the step that should
win back the non-stationary setting UCB1 is about to lose.
