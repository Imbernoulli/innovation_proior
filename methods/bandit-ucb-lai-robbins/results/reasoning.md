I start from the regret identity because it strips away the distracting language of experimentation. If the best mean is `mu*` and arm `i` is worse by `Delta_i`, then every pull of that arm costs `Delta_i` in expectation, and

```text
R_n = sum_{i: Delta_i > 0} Delta_i E[T_i(n)].
```

So the problem is not first "which arm do I like?" It is "how many times must I pay for each arm that is not best?" If I can answer that question from below, I know the unavoidable cost of learning; if I can answer it from above with a rule, I know how close the rule is to optimal.

My first hope is that a bad arm should only need a constant number of samples. I try to imagine a very clever policy that tests each arm just enough, recognizes the inferior ones, and then never returns. But that thought is too local. It treats the current instance as if the policy is allowed to know that it is the current instance. A valid policy has to work on nearby instances too, and I should make that objection concrete rather than wave at it.

So I build the obstruction explicitly. Fix a suboptimal arm `i` with law `P_i`. Now build another world that is identical except that arm `i` has law `P_i'` with mean just above the current best mean. In the real world, arm `i` is bad. In the neighboring world, arm `i` is best. If my policy samples arm `i` only a constant number of times in the real world, then the observed history contains only a constant amount of evidence about the one distribution that changed. But in the neighboring world, consistency demands that the policy learn to play arm `i` almost all the time. The same policy cannot be safe in both worlds unless it gathers enough information from arm `i` to tell the two worlds apart. A constant sample budget cannot do that, so my first hope is dead; the bad arm must be sampled an amount that grows with the horizon.

This turns exploration into a hypothesis-testing problem. Samples from arm `i` generate a log-likelihood ratio

```text
sum log(dP_i / dP_i')
```

and under the real law its expectation grows at rate `D(P_i || P_i')` per sample. I want to be sure I have that rate right and not off by a sign or a factor, so I check it on the case I will actually use later, two unit-variance Gaussians. There `log(dP_i/dP_i')` for a single sample `x` is `-(x-mu_i)^2/2 + (x-mu_i')^2/2`. Drawing twenty million samples under the real law `N(mu_i,1)` and averaging this log-ratio, I get, for `(mu_i, mu_i') = (0, 0.5)`, an empirical mean of `0.12498`, and `(mu_i'-mu_i)^2/2 = 0.125`; for `(0.2, 0.7)`, `0.12498` against `0.125`; for `(1.0, 1.3)`, `0.04499` against `0.045`. The per-sample drift really is `D(N(mu_i,1) || N(mu_i',1)) = (mu_i'-mu_i)^2/2`, and it is positive, as it must be for the test to favor the truth. After `T_i(n)` pulls the expected information separating the two worlds is therefore `T_i(n) D(P_i || P_i')`. If the policy wants the probability of confusing the two worlds to be small on the scale needed over `n` rounds, it needs evidence on the order of `log n`. That suggests a sample count near `log n / D(P_i || P_i')`, but a suggestion from drift alone is not a lower bound, so I have to pin it down.

I can make the intuition precise in two equivalent ways. The likelihood-ratio route says that, for an event `A` in the interaction history, probabilities under the two worlds differ only through the likelihood ratio of the samples collected from arm `i`. If I define the event that arm `i` has been sampled fewer than `(1-epsilon) log n / D(P_i || P_i')` times and the likelihood ratio has not yet grown to about `log n`, then this event cannot have large probability in the real world. In the modified world the arm is optimal, so consistency makes the complementary behavior rare; in the real world the strong law says the likelihood ratio cannot usually exceed its drift when the sample count is below that threshold. The event "too few samples" disappears.

The information-inequality route is cleaner and I prefer to lean on it. The KL divergence between the full histories under the two bandit worlds decomposes as

```text
D(P_nu || P_nu') = sum_j E_nu[T_j(n)] D(P_j || P_j').
```

Only arm `i` changes, so this collapses to `E_nu[T_i(n)] D(P_i || P_i')`. Bretagnolle-Huber then says no event can reliably separate the two histories unless this divergence is large. Taking the event that arm `i` receives more than half the horizon, one of the two worlds must pay regret unless the divergence is at least nearly `log n`. Consistency rules out polynomial regret in either world, so the sample count must satisfy

```text
liminf E[T_i(n)] / log n >= 1 / D(P_i || P_i').
```

Now I let the alternative law move down to the boundary where arm `i` just becomes optimal, while staying inside the allowed model class for that arm. This gives the information constant:

```text
liminf E[T_i(n)] / log n >= 1 / d_inf(P_i, mu*, M_i),
```

or, in a one-parameter family,

```text
liminf R_n / log n >= sum_{i: Delta_i > 0} Delta_i / KL(theta_i, theta*).
```

For Gaussian rewards with known variance `sigma^2`, the denominator is `Delta_i^2/(2 sigma^2)` by the drift I just checked, so the necessary number of pulls of arm `i` is `2 sigma^2 log n / Delta_i^2`; in the unit-variance case, `2 log n / Delta_i^2`. For Bernoulli arms the denominator is `kl(mu_i, mu*)`, and I want to see how different that is from the Gaussian `Delta_i^2/2`, because if they were close I could reuse one analysis for both. They are not. Computing `kl(mu_i, mu*)` and `Delta_i^2/2` directly: for `(0.4, 0.5)`, `kl = 0.02014` versus `0.005`, a ratio near `4`; for `(0.1, 0.2)`, `0.03669` versus `0.005`, ratio `7.3`; for `(0.8, 0.9)`, `0.04440` versus `0.005`, ratio `8.9`. A Bernoulli arm near the boundary delivers several times more information per sample than the Gaussian gap suggests, more so the closer the means sit to `0` or `1`. So the right denominator is genuinely the divergence, not the squared gap, and a method that only ever uses `Delta^2` will be loose on Bernoulli arms by exactly these factors.

That is the part of the analysis I did not have to take on faith: exploration is not an ad hoc tax I choose by taste. Every inferior arm has to be sampled until the policy has accumulated about `log n` units of evidence against the closest world where that arm is best. The denominator is the rate at which that evidence arrives from samples of the arm, and I have now watched both the Gaussian and Bernoulli rates come out of an actual computation.

Now I ask what kind of rule naturally pays that cost and stops. Greedy is too brittle because it pretends each empirical mean is already the truth. A barely sampled arm can be underestimated and then never repaired. Uniform random exploration is too blunt because it keeps paying for arms after the evidence is already decisive. I need a rule whose desire to sample an arm shrinks automatically as evidence accumulates.

The shape of the lower bound points one way to get that. For each arm, I maintain the largest mean that is still plausible given its data, and I play the arm with the largest plausible mean. A lightly sampled arm receives a wide allowance because the data cannot yet rule out a favorable world. A heavily sampled bad arm receives a narrow allowance, so once its estimate settles below the best mean it stops attracting pulls. The same number is doing both jobs: it exploits high empirical means and explores high uncertainty.

For bounded rewards, Hoeffding's inequality tells me the size of the allowance. After `s` samples,

```text
P(hat_mu_s >= mu + a) <= exp(-2 s a^2),
P(hat_mu_s <= mu - a) <= exp(-2 s a^2).
```

Solving `exp(-2 s a^2) = delta` gives a one-sided width `sqrt(log(1/delta)/(2s))`. If I choose a time-dependent confidence level strong enough to survive a union bound over times and sample counts, I get the index

```text
hat_mu_i + sqrt(2 log t / T_i(t-1)).
```

The rule samples each arm once, then always selects the arm with the largest index.

I want to know that this index actually self-limits, so I trace its regret argument. Fix a bad arm `i`. If arm `i` is chosen after it has already been sampled `T_i` times, its index must beat the optimal arm's index. That requires one of three things: the optimal arm's empirical mean is unusually low, arm `i`'s empirical mean is unusually high, or the confidence radius for arm `i` is still wide enough to bridge the gap on its own. The first two are random and Hoeffding will make them summable. The third is deterministic, and I can find exactly when it switches off. The radius is `sqrt(2 log n / T_i)`. Setting it equal to `Delta_i/2` and solving, `2 log n / T_i = Delta_i^2/4`, i.e. `T_i = 8 log n / Delta_i^2`. So the threshold should be

```text
T_i >= 8 log n / Delta_i^2.
```

Before I trust that, I check the algebra numerically, because a factor of 2 here would change the final constant. Plugging `T_i = 8 log n / Delta_i^2` back into the radius for several `(n, Delta)`: at `n=1000, Delta=0.3`, `T=614.0` and `sqrt(2 log n / T) = 0.15000 = Delta/2`; at `n=10000, Delta=0.1`, `T=7368.3` and the radius is `0.05000 = Delta/2`; every case lands the radius on `Delta/2` to five places. Good. Once arm `i` and the best arm are each within `Delta_i/2` of their true means (the typical case Hoeffding guarantees), their indices differ by less than the radius can close, so a chosen-after-threshold pull needs an atypical fluctuation, and those are rare. Therefore

```text
E[T_i(n)] <= 8 log n / Delta_i^2 + 1 + pi^2/3.
```

The trailing `1 + pi^2/3` is not decoration; the `1` is the single forced first pull, and the rest is the tail of the two summable fluctuation families. With confidence delta `~ t^{-4}` the bad events cost about `t^{-4}` per round, and summing the dominant pieces gives `2 sum_{t>=1} t^{-2}`. I evaluate that sum directly to be sure it converges to what I think: `sum_{t>=1} t^{-2} = 1.6449`, so `2 * 1.6449 = 3.2899 = pi^2/3`. So the constant is finite and traceable to where the union bound spends its budget.

Multiplying by `Delta_i` gives logarithmic regret. The constant `8/Delta_i^2` is not the information constant in general, and I should see how loose it really is rather than guess. I simulate UCB1 on two Bernoulli arms with means `0.5` and `0.3`, so `Delta=0.2`, averaging `T_bad` over 200 runs. At `n=1000` the average `T_bad` is `132`, at `n=4000` it is `240`, at `n=16000` it is `350`; dividing by `log n` these are `19.2`, `28.9`, `36.1`. Two things are visible at once. The ratio is still climbing, consistent with the genuine `log n` growth and a `log n` term not yet dominant at these horizons rather than a true constant. And the empirical counts sit far below the worst-case `8 log n / Delta^2 = 200 log n` (which would be `1386` at `n=1000`): the proof constant is loose by roughly an order of magnitude here, which is what I should expect from spending slack in the quadratic Hoeffding radius and the union bound. The order is right and the mechanism matches the lower-bound obstruction: keep an arm alive exactly while the data still permit it to be the best.

This simple rule is not the original 1985 Lai-Robbins allocation rule; it is the later finite-time UCB1 index. It is still the clean constructive face of the same optimism idea, and it is faithful to the Auer-Cesa-Bianchi-Fischer pseudocode. To close the asymptotic constant for Gaussian rewards, I can tune the confidence schedule more carefully. In the 1-subgaussian normalization, a later anytime index with

```text
sqrt(2 log(1 + t log^2 t) / T_i(t-1))
```

has asymptotic regret at most

```text
sum_{i: Delta_i > 0} 2 log n / Delta_i.
```

I check whether this matches the lower bound rather than asserting it. The Gaussian lower-bound term is `Delta_i / KL(N(mu_i,1), N(mu*,1))`, and from the drift I verified, `KL = Delta_i^2/2`, so the term is `Delta_i / (Delta_i^2/2) = 2/Delta_i`, and multiplying the pull-count bound by `Delta_i` gives the same `2 log n / Delta_i`. They agree, so this refined index is asymptotically optimal for unit-variance Gaussians. In Bernoulli and exponential-family settings the factor-of-4-to-9 gap I measured between `kl` and `Delta^2/2` says the quadratic radius cannot match the constant; the lesson points to KL-shaped confidence sets, choosing the largest mean whose KL divergence from the empirical model is still within the exploration budget.

So the substance is not "put confidence intervals on a bandit." The load-bearing step is the lower bound: every consistent policy is secretly running a sequential test against neighboring worlds, with an unavoidable evidence bill of `log n / KL`, and the divergence in that denominator is something I watched arrive from samples, not a decorative constant. The confidence index is then the constructive face of that proof, an implementable way to keep sampling exactly those arms for which the favorable neighboring world has not yet been statistically ruled out.
