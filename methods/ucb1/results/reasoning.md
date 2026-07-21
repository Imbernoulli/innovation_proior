Let me start from what actually goes wrong. I have `K` arms, each returning i.i.d. rewards in `[0,1]` with an unknown mean `mu_i`. Every round I pick one arm, see its reward, and that's all I see — nothing about the arms I didn't pull. I want to do nearly as well as always playing the best arm, so the thing I'm really fighting is regret: with `mu* = max_i mu_i` and `T_i(n)` the number of times I've pulled arm `i` in `n` rounds, the loss is `mu* n - sum_j mu_j E[T_j(n)]`. Rearrange that — `mu* n = sum_j mu* E[T_j(n)]` since the pull counts sum to `n` — and it collapses to `R_n = sum_{j: mu_j < mu*} Delta_j E[T_j(n)]` with `Delta_j = mu* - mu_j`. So regret is literally a sum over the bad arms of (how far below the best each one is) times (how many times I pull it). The means and gaps are fixed by the world; the only thing my policy controls is `E[T_j(n)]`, the expected number of pulls of each suboptimal arm. That's the whole game: pull the bad arms as few times as I can get away with.

The obvious thing is to estimate each mean by its empirical average `x_bar_j = (cumulative reward of j) / N_j` and always play the largest. But I can see this die. Suppose the best arm has the highest mean but, on its first couple of pulls, returns unlucky low rewards — `[0,1]` rewards are noisy, this happens. Its empirical mean drops below some mediocre arm's, the greedy rule stops choosing it, and here's the trap: because I never pull it again, its estimate never gets another sample, so it stays buried forever. I lock onto a worse arm and pay its gap on every remaining round — regret `Theta(n)`, linear, the worst possible. The lesson is sharp: an arm I've sampled few times, I genuinely don't know about, and I can't let a noisy estimate permanently exile it. I need to keep some willingness to revisit under-sampled arms.

How much willingness, though? The crude fix is to flip a coin: with probability `1 - eps` play the empirical best, with probability `eps` play a uniformly random arm — `eps`-greedy. That does break the lock-in, but think about the cost. A constant `eps` means a constant fraction of *all* rounds are uniformly random forever, and a random pull lands on a bad arm with constant probability, so I'm paying gap-sized regret on `~ eps n` rounds — still linear in `n`. To get sublinear regret I'd have to make `eps` shrink as I learn — `eps_n` decreasing — but then I have to tune the decay rate, and tuning it correctly to guarantee a logarithmic bound turns out to need a lower bound on the smallest gap `Delta`, which I don't know a priori. So undirected exploration — exploring by coin flip, uniformly, blind to which arms are actually uncertain — wastes pulls on arms I'm already sure are bad. I want exploration that is *targeted*: spend exploratory pulls on the arms whose value I'm genuinely unsure about, and stop spending them once I'm sure. The uncertainty itself should decide where to explore.

So I need, per arm, not just a point estimate `x_bar_j` but a sense of how *trustworthy* that estimate is. An arm pulled a thousand times — I know its mean well. An arm pulled twice — I barely know it; its true mean could plausibly be much higher than the two samples suggest. What if, instead of comparing arms by their empirical means, I compare them by an *optimistic* estimate — for each arm, the highest value its true mean could plausibly take given the data — and play the largest of those? Picture what this does. A genuinely good, well-sampled arm has a tight estimate near its true (high) mean, so its optimistic value is high — I exploit it. A genuinely bad, well-sampled arm has a tight estimate near its true (low) mean, so its optimistic value is low — I leave it alone, correctly. The interesting case is an under-sampled arm: its estimate is loose, so its optimistic value is inflated well above its empirical mean — which pulls me toward sampling it. And if that optimism was misplaced (the arm really is bad), pulling it tightens its estimate and the inflated optimistic value collapses down toward the true low mean, so I quickly stop. But if the optimism was warranted (the arm really is good and I'd under-sampled it), I keep pulling and reap it.

Check that this actually defuses the burying scenario that killed pure greedy. Take the same setup: the *best* arm gets an unlucky reward of `0` on its one early pull. Concretely, two arms, and suppose I've now pulled each once, arm `0` returning `0` and arm `1` returning `1`, so we're at total time `t = 2`. Under pure greedy I'd compare empirical means `0` vs `1` and never touch arm `0` again. Under the optimistic rule I compare `x_bar_j + sqrt(2 ln t / N_j)` with `t = 2`, `N_j = 1` each. The bonus is `sqrt(2 ln 2) = sqrt(1.386) = 1.177` for both. So arm `0`'s index is `0 + 1.177 = 1.177` and arm `1`'s is `1 + 1.177 = 2.177`. Arm `1` wins this round — but the decisive thing is that arm `0`'s index is `1.177`, *not* `0`. It is not buried; the bonus keeps the unlucky arm in contention, and as `ln t` creeps up its inflated index can climb back above arm `1`'s and it gets re-sampled. The flip side is just as concrete: if I keep pulling a genuinely bad arm, its `N_j` grows and `sqrt(2 ln t / N_j)` shrinks faster than `ln t` grows, so its inflated index decays back toward its true low mean and I stop. Either way the policy self-corrects: a wrongly-high optimistic value cannot survive being acted on. That's the property `eps`-greedy lacked — exploration directed by uncertainty, and self-extinguishing when the uncertainty is resolved the wrong way. So the design rule I'll commit to is: be optimistic in the face of uncertainty, then act greedily on the optimism — play the arm with the largest plausible mean. This is the upper-confidence-index idea, the same shape Lai and Robbins used for their asymptotically optimal rules and that Agrawal made cheap to compute by writing the index as a simple function of the rewards seen so far. What neither pinned down for me is the precise radius that yields a bound holding at every finite `n`, not just in the limit — so that is what I still have to derive.

Now I have to make "the highest plausible value of `mu_j`" precise, and that's a confidence interval. I need: how far above the empirical mean `x_bar_j` could the true mean `mu_j` be, with overwhelming probability, given `N_j` samples? Concentration answers exactly this. The Chernoff-Hoeffding bound says that for `s` i.i.d. (more generally, conditionally-mean-`mu`) samples in `[0,1]`, the empirical mean concentrates around `mu` with sub-Gaussian-style tails: `P{ x_bar >= mu + a } <= e^{-2 s a^2 }` and symmetrically `P{ x_bar <= mu - a } <= e^{-2 s a^2 }`. So if I want a one-sided confidence statement that fails with probability at most some small `delta`, I set `e^{-2 s a^2} = delta`, i.e. `2 s a^2 = ln(1/delta)`, i.e. the radius

```
a = sqrt( ln(1/delta) / (2 s) ).
```

That's the half-width of the interval. The upper confidence bound on `mu_j` is then `x_bar_j + a`, and I'll play the arm maximizing it. The radius already has the two qualitative behaviors I wanted, just from its form: it shrinks like `1/sqrt(s)` as I pull an arm more (its optimistic value descends toward the truth — exploit good arms, abandon bad ones), and — if I let `delta` depend on time so `ln(1/delta)` grows — it can creep back up for arms I've stopped pulling, so no arm is dismissed forever. The question is what to pick for `delta`.

Here's the tension. I want `delta` tiny, so the confidence intervals essentially never fail (a failure means my optimism was *under*-estimating the best arm, which is how I'd wrongly abandon it). But I'm going to apply this interval at every round and for every arm and for every possible sample count, so when I bound the total probability of *any* failure over the whole run with a union bound, I'll be summing a lot of `delta`'s. If `delta` is a constant the sum diverges. So `delta` must shrink with time fast enough that the union-bounded total stays finite. Let me tie `delta` to the round index. At round `t` I've made `t` plays total. If I set `delta = t^{-4}`, then `ln(1/delta) = 4 ln t` and the radius becomes

```
a = sqrt( 4 ln t / (2 s) ) = sqrt( 2 ln t / s ).
```

Let me write `c_{t,s} = sqrt( 2 ln t / s )` for the confidence radius at round `t` for an arm pulled `s` times. Why `t^{-4}` and not, say, `t^{-2}`? I can't justify the exponent yet — it has to come out of making the regret proof's union bound converge — so let me carry it as a choice to validate, write the algorithm, and see what the proof demands.

The policy, then: first play each arm once (so every `N_j >= 1` and the radius `sqrt(2 ln t / N_j)` is well-defined — no division by zero). After that, at each round play the arm `j` maximizing

```
x_bar_j + sqrt( 2 ln n / N_j ),
```

where `x_bar_j` is the current empirical mean of arm `j`, `N_j` its pull count, and `n` the total number of plays so far. Empirical mean plus an uncertainty bonus; argmax; that's it. It's cheap — each round I just recompute `K` indices from running sums — which is the practicality Lai and Robbins' general index lacked, and it assumes nothing about the reward distributions beyond their support in `[0,1]`.

Now the real work: does this actually keep `E[T_j(n)]` logarithmic, with a bound I can write down for every finite `n`? Let me try to bound, for a fixed suboptimal arm `i`, the number of times it gets pulled. I'll think of it as: arm `i` only gets pulled at round `t` if, at that moment, its index beat the optimal arm's index. So `i` being over-pulled means its index keeps winning, and I want to show that can't happen too often. Let me set up the counting carefully. Let `c_{t,s} = sqrt(2 ln t / s)`, let the optimal arm carry a star, and let `ell` be some threshold count for arm `i` that I'll choose later. Writing `1{...}` for the indicator of an event,

```
T_i(n) = 1 + sum_{t=K+1}^n 1{ I_t = i }
```

(the `1` is its one forced pull in the initialization). Now I'll only count the pulls after `i` has already been pulled at least `ell` times — the first `ell` I just concede:

```
T_i(n) <= ell + sum_{t=K+1}^n 1{ I_t = i  and  T_i(t-1) >= ell }.
```

When does `I_t = i` happen? Only if `i`'s index is at least the optimal arm's index at round `t`. At that round the optimal arm has been pulled `T*(t-1)` times and arm `i` has been pulled `T_i(t-1)` times, so the event `I_t = i` forces

```
x_bar*_{T*(t-1)} + c_{t-1, T*(t-1)} <= x_bar_{i, T_i(t-1)} + c_{t-1, T_i(t-1)}.
```

I don't know the actual sample counts `T*(t-1)` and `T_i(t-1)`, but whatever they are, the optimal arm's side is at least its minimum over all plausible counts, and arm `i`'s side is at most its maximum over plausible counts (with `i`'s count at least `ell`). So this is implied by

```
min_{0 < s < t} ( x_bar*_s + c_{t,s} )  <=  max_{ell <= s_i < t} ( x_bar_{i, s_i} + c_{t, s_i} ).
```

(I bumped `t-1` up to `t` inside the radii; that only loosens things, since `c` increases in its first argument.) And a min being `<=` a max means *some* pair `(s, s_i)` realizes the inequality, so I can drop the min/max and union-bound over all sample counts:

```
T_i(n) <= ell + sum_{t=1}^{n} sum_{s=1}^{t-1} sum_{s_i = ell}^{t-1} 1{ x_bar*_s + c_{t,s} <= x_bar_{i, s_i} + c_{t, s_i} }.
```

I also widened the lower limit from `K+1` down to `1`, again only adding nonnegative terms. I keep the outer sum capped at `n`, not further — I'll need that cap in a moment to kill event (9) uniformly, and I can loosen the resulting *numeric* series to a closed form only after that's done. Now I have a clean triple sum of indicator events, and I need to bound the probability of each event `{ x_bar*_s + c_{t,s} <= x_bar_{i,s_i} + c_{t,s_i} }`.

Stare at that event. It says the optimal arm's optimistic value, computed from `s` samples, came out *no larger* than suboptimal arm `i`'s optimistic value, computed from `s_i` samples. For that to happen, at least one of three things must have gone wrong, and this trichotomy is exhaustive. The three candidate failures are:

```
(7)  x_bar*_s <= mu* - c_{t,s}            -- the optimal arm's empirical mean fell a full radius below its true mean (bad luck against me),
(8)  x_bar_{i,s_i} >= mu_i + c_{t,s_i}    -- arm i's empirical mean rose a full radius above its true mean (bad luck for me),
(9)  mu* < mu_i + 2 c_{t,s_i}            -- the confidence radius for arm i is so wide that even with no estimation error its optimistic value could top mu*.
```

Suppose *none* of (7), (8), (9) holds. Then `x_bar*_s > mu* - c_{t,s}`, so `x_bar*_s + c_{t,s} > mu*`. And `mu* >= mu_i + 2 c_{t,s_i}` (negation of (9)), and `x_bar_{i,s_i} < mu_i + c_{t,s_i}` (negation of (8)), so `x_bar_{i,s_i} + c_{t,s_i} < mu_i + 2 c_{t,s_i} <= mu*`. Chaining: `x_bar*_s + c_{t,s} > mu* > x_bar_{i,s_i} + c_{t,s_i}`, which says the optimal arm's optimistic value is *strictly greater* than `i`'s — the negation of the event. So the event indeed implies at least one of (7), (8), (9). The trichotomy holds.

Now I get to choose `ell` to kill (9) outright. Event (9) is `mu* - mu_i < 2 c_{t,s_i}`, i.e. `Delta_i < 2 sqrt(2 ln t / s_i)`. This is a deterministic statement about `s_i` — no randomness — so if I force `s_i` large enough it simply can't hold. Solve `Delta_i >= 2 sqrt(2 ln t / s_i)`: square it, `Delta_i^2 >= 8 ln t / s_i`, i.e. `s_i >= 8 ln t / Delta_i^2`. Since I only sum over `t <= n`, I have `ln t <= ln n`, so `s_i >= 8 ln n / Delta_i^2` guarantees `s_i >= 8 ln t / Delta_i^2` for every `t` in range, which makes (9) false. So set

```
ell = ceil( 8 ln n / Delta_i^2 ).
```

This constant `8` isn't a tuning choice, it's forced: the radius carries the Hoeffding `2` in its numerator, I need *twice* the radius to be at most `Delta_i` (one radius of slack for each arm's side of the comparison), and `(2 * sqrt(2))^2 = 8`. A more cautious (wider) radius would have forced a bigger number here; `8 ln n/Delta_i^2` is exactly the exploration count past which arm `i`'s confidence interval is too tight to plausibly beat the optimum.

With (9) gone, only (7) and (8) remain, and those are the genuinely probabilistic events — exactly the ones Hoeffding bounds. Take (7): `x_bar*_s <= mu* - c_{t,s}`. By Chernoff-Hoeffding with `s` samples and radius `c_{t,s} = sqrt(2 ln t / s)`,

```
P{ x_bar*_s <= mu* - c_{t,s} } <= e^{-2 s c_{t,s}^2 } = e^{-2 s (2 ln t / s)} = e^{-4 ln t} = t^{-4}.
```

And there's the payoff of the `t^{-4}` choice of `delta`: the `2` in the exponent (Hoeffding) times the `2` in the radius's numerator (which came from `ln(1/delta) = 4 ln t`) gives `4 ln t`, so each failure event has probability `t^{-4}`. Symmetrically for (8): `P{ x_bar_{i,s_i} >= mu_i + c_{t,s_i} } <= e^{-2 s_i c_{t,s_i}^2 } = t^{-4}`. Both bad events, probability `t^{-4}` each.

Now sum. Taking expectations of the triple-sum bound and using `ell = ceil(8 ln n/Delta_i^2)` so that (9) never contributes,

```
E[T_i(n)] <= ceil(8 ln n / Delta_i^2) + sum_{t=1}^{n} sum_{s=1}^{t-1} sum_{s_i = ceil(8 ln n/Delta_i^2)}^{t-1} ( P(7) + P(8) ).
```

Bound `P(7) + P(8) <= 2 t^{-4}`, and overcount the inner double sum: for each `t`, the indices `s` and `s_i` each run over fewer than `t` values, so the double sum has at most `t * t = t^2` terms, giving `<= t^2 * 2 t^{-4} = 2 t^{-2}`. Summing `2 t^{-2}` over `t = 1, ..., n` gives a finite total; now, and only now, is it safe to loosen that finite sum to the full convergent series, since replacing a partial sum of a positive series by the whole series can only make it bigger. So

```
E[T_i(n)] <= 8 ln n / Delta_i^2 + 1 + sum_{t=1}^{n} 2 t^{-2} <= 8 ln n / Delta_i^2 + 1 + sum_{t=1}^{infty} 2 t^{-2} = 8 ln n / Delta_i^2 + 1 + 2 * (pi^2 / 6) = 8 ln n / Delta_i^2 + 1 + pi^2/3,
```

using `ceil(x) <= x + 1` and the Basel sum `sum_{t>=1} 1/t^2 = pi^2/6`. The expected number of pulls of suboptimal arm `i` is `8 ln n / Delta_i^2`, plus a *constant* — the `1 + pi^2/3` is independent of `n`. This is exactly why `delta = t^{-4}` and not something milder: the union bound left me with `~ t^2` events at probability `delta` each, and I needed `t^2 * delta` to sum over `t` to a finite constant; `delta = t^{-4}` gives `t^{-2}`, whose sum converges. A milder `delta = t^{-2}` would have given `t^2 * t^{-2} = 1` per round, summing to infinity — the additive cost would grow linearly in `n` and the bound would collapse. The exponent `4` is the smallest clean choice that makes the post-union-bound series converge.

Convert pull counts to regret. Regret is `R_n = sum_{i: mu_i < mu*} Delta_i E[T_i(n)]`. Multiply the per-arm bound by `Delta_i`: the leading term `Delta_i * 8 ln n / Delta_i^2 = 8 ln n / Delta_i`, and the constant term `Delta_i * (1 + pi^2/3)`. Summing,

```
R_n <= 8 sum_{i: mu_i < mu*} (ln n / Delta_i) + (1 + pi^2/3) sum_{j=1}^K Delta_j.
```

That's the finite-time regret bound, and it holds for every finite horizon, not just asymptotically — which is exactly the hole the prior art left. The leading term grows like `ln n` (times a sum of `1/Delta_i` over bad arms), and the rest is a fixed constant set by the gaps. Logarithmic regret, uniform over time, distribution-free over `[0,1]`. The targeted-optimism design paid off: a bad arm gets pulled only about `8 ln n / Delta_i^2` times, because after that many samples its confidence interval is too tight to keep beating the optimum, and the union bound shows the rare confidence failures only ever cost a bounded constant on top.

Two readings of this bound deserve a moment. First, against the floor. Lai and Robbins say the unavoidable rate is `(ln n) / D(p_i || p*)` pulls of a bad arm, governed by KL divergence to the optimal arm. My bound has `8 ln n / Delta_i^2` — the right `ln n` *order*, but what about the constant? For bounded rewards a Pinsker-type inequality gives `D(p_i || p*) >= 2 Delta_i^2`, with the constant `2` known to be tight as `Delta_i -> 0` — checked on Bernoulli arms at a small gap, `p* = 0.6, p_i = 0.5`: the binary KL `d(p_i, p*) = p_i ln(p_i/p*) + (1-p_i) ln((1-p_i)/(1-p*)) = 0.0204`, against `2 Delta^2 = 0.02`, ratio `1.02`, essentially equality. So the floor allows as few as `(ln n)/(2 Delta_i^2)` pulls, while I'm spending `8 ln n/Delta_i^2` — a factor of about `16` above the information-theoretic minimum. I have the order but a loose constant, and the looseness traces straight back to the `8`, which traced back to needing `2 c_{t,s_i} <= Delta_i` with the Hoeffding radius. If I made the radius adapt by epochs and shaved the slack, I could chase the constant down toward `1/(2 Delta_i^2)` — and indeed a slightly more elaborate epoch-based variant (play each arm in geometrically growing batches, with radius `sqrt((1+alpha) ln(e n / tau(r)) / (2 tau(r)))` over the `r`-th epoch of length `tau(r) = ceil((1+alpha)^r)`) brings the leading constant arbitrarily close to `1/(2 Delta_i^2)` as `alpha -> 0`, at the price of a constant that blows up as `alpha -> 0`, so the two terms trade off and you let `alpha` decay slowly with `n`. But for the simple rule, `8` it is, and the order is what matters most.

Second reading: the gap-free, worst-case view. The bound `8 ln n / Delta_i` is great when `Delta_i` is a healthy constant, but it *explodes* as `Delta_i -> 0` — a nearly-tied arm has a huge `1/Delta_i`. That cannot be the real cost, though, because a nearly-tied arm is barely suboptimal: pulling it costs almost nothing per pull. I should not balance the two bounds separately for every arm, because that would throw away the fact that the total number of pulls over all small-gap arms is only `n`. Let me split the arms at a threshold `gamma`. Arms with `Delta_i <= gamma` contribute at most `gamma` per pull and there are only `n` pulls total, so their total regret is at most `n gamma`. Arms with `Delta_i > gamma` can use the logarithmic bound, and since `1/Delta_i < 1/gamma`, their leading contribution is at most `8 K ln n / gamma`, plus the additive constants from the theorem. Thus

```
R_n <= n gamma + 8 K ln n / gamma + constant terms.
```

Balancing the two leading pieces — set `d/d gamma (n gamma + 8 K ln n / gamma) = n - 8 K ln n / gamma^2 = 0` — gives `gamma = sqrt(8 K ln n / n)`. Substituting back, both pieces become `n * sqrt(8 K ln n / n) = sqrt(8 K n ln n)`, so the sum is `2 sqrt(8 K n ln n) = 2 sqrt(8) sqrt(K n ln n) ~ 5.66 sqrt(K n ln n)`, i.e. `O(sqrt(K n ln n))`. So the same theorem, read without committing to a particular gap profile, gives a minimax-style sublinear guarantee and therefore average regret `R_n / n -> 0`.

Worth noting how little the proof actually leaned on: Hoeffding only needed `[0,1]` range and the conditional-mean property, not full independence — the bound holds under `E[X_{i,t} | X_{i,1}, ..., X_{i,t-1}] = mu_i` alone, with no requirement of independence across arms either, so it survives mild dependence and non-i.i.d. rewards within an arm.

One more thing I should square away before coding: the algorithm's index uses `n` (total plays so far) inside the log, while the proof wrote `c_{t,s} = sqrt(2 ln t / s)` with `t` the round index. At the moment I'm choosing the next arm, the policy's clock is exactly the number of rewards already received, so the log is the global time-so-far. The implementation can encode the "play each arm once" initialization by returning `+inf` for any arm with zero pulls; then the generic index-policy argmax will keep choosing unpulled arms until all indices are finite. If there are several `+inf` arms, a random tie-break merely randomizes the order of the first `K` pulls. After that, the bounded-reward index is exactly empirical mean plus `sqrt(2 log(t) / N_j)`.

Now, what about the case where the reward range is not a known `[0,1]` interval but the arms are Gaussian with unknown mean *and* unknown variance? Then the relevant scale of fluctuation is the variance, not a fixed range, and the Hoeffding radius `sqrt(2 ln t / s)` — which only knows the range — is the wrong width. The fix mirrors the main construction: estimate the variance from the samples and size the confidence interval by it. With the sample sum of squares `q_j` and `N_j` pulls, the unbiased sample variance is `(q_j - N_j x_bar_j^2)/(N_j - 1)`, and the index becomes `x_bar_j + sqrt( 16 * (q_j - N_j x_bar_j^2)/(N_j - 1) * ln(n-1) / N_j )`, paired with a rule to play any arm that has been pulled fewer than `ceil(8 log n)` times so the variance estimate is not based on too little data. The proof no longer rests on Hoeffding alone; it needs tail bounds for chi-squared and Student statistics, and those bounds have to be taken as numerically checked analytic facts. Under those tail bounds the same style of theorem gives regret `<= 256 (log n) sum_i sigma_i^2/Delta_i + (1 + pi^2/2 + 8 log n) sum_j Delta_j` — the same `log n` order, with the per-arm cost scaled by the arm's variance `sigma_i^2` rather than the worst-case `[0,1]` range.

The state needed to run this is exactly the running sufficient statistics the given scaffold already tracks: per-arm pull counts, per-arm cumulative rewards, and a global clock `t`. `computeIndex(arm)` returns `+inf` while `pulls[arm] < 1` (which forces the initial one-pull-per-arm round-robin through the generic argmax), and otherwise the empirical mean plus the bonus `sqrt(2 log(t) / N_j)` derived above. It's convenient to generalize the bonus to `sqrt(alpha * log(t) / (2 N_j))` for a free constant `alpha`, since `alpha = 4` collapses to exactly this rule while other choices trade off the same confidence-radius shape against a different exploration rate — this is the same upper-confidence-index template Lai-Robbins and Agrawal used, just with the specific radius pinned down here; the full module is in the answer.
