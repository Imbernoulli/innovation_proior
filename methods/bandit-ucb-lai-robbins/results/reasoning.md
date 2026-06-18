I start from the regret identity because it strips away the distracting language of experimentation. If the best mean is `mu*` and arm `i` is worse by `Delta_i`, then every pull of that arm costs `Delta_i` in expectation, and

```text
R_n = sum_{i: Delta_i > 0} Delta_i E[T_i(n)].
```

So the problem is not first "which arm do I like?" It is "how many times must I pay for each arm that is not best?" If I can answer that question from below, I know the unavoidable cost of learning; if I can answer it from above with a rule, I know how close the rule is to optimal.

My first hope is that a bad arm should only need a constant number of samples. I try to imagine a very clever policy that tests each arm just enough, recognizes the inferior ones, and then never returns. But that thought is too local. It treats the current instance as if the policy is allowed to know that it is the current instance. A valid policy has to work on nearby instances too.

The obstruction is a neighboring world. Fix a suboptimal arm `i` with law `P_i`. Now build another world that is identical except that arm `i` has law `P_i'` with mean just above the current best mean. In the real world, arm `i` is bad. In the neighboring world, arm `i` is best. If my policy samples arm `i` only a few times in the real world, then the observed history contains only a few pieces of evidence about the one distribution that changed. But in the neighboring world, consistency demands that the policy learn to play arm `i` almost all the time. The same policy cannot be safe in both worlds unless it gathers enough information from arm `i` to tell the two worlds apart.

This turns exploration into a hypothesis-testing problem. Samples from arm `i` generate a log-likelihood ratio

```text
sum log(dP_i / dP_i')
```

and under the real law its expectation grows at rate `D(P_i || P_i')` per sample. After `T_i(n)` pulls, the expected information separating the two worlds is therefore `T_i(n) D(P_i || P_i')`. If the policy wants the probability of confusing the two worlds to be small on the scale needed over `n` rounds, it needs evidence on the order of `log n`. That already suggests

```text
T_i(n) about log n / D(P_i || P_i').
```

I can make the intuition precise in two equivalent ways. The likelihood-ratio route says that, for an event `A` in the interaction history, probabilities under the two worlds differ only through the likelihood ratio of the samples collected from arm `i`. If I define the event that arm `i` has been sampled fewer than `(1-epsilon) log n / D(P_i || P_i')` times and the likelihood ratio has not yet grown to about `log n`, then this event cannot have large probability in the real world. In the modified world the arm is optimal, so consistency makes the complementary behavior rare; in the real world the strong law says the likelihood ratio cannot usually exceed its drift when the sample count is below that threshold. The event "too few samples" disappears.

The information-inequality route is cleaner. The KL divergence between the full histories under the two bandit worlds decomposes as

```text
D(P_nu || P_nu') = sum_j E_nu[T_j(n)] D(P_j || P_j').
```

Only arm `i` changes, so this becomes `E_nu[T_i(n)] D(P_i || P_i')`. Bretagnolle-Huber then says no event can reliably separate the two histories unless this divergence is large. Taking the event that arm `i` receives more than half the horizon, one of the two worlds must pay regret unless the divergence is at least nearly `log n`. Consistency rules out polynomial regret in either world, so the sample count must satisfy

```text
liminf E[T_i(n)] / log n >= 1 / D(P_i || P_i').
```

Now I let the alternative law move down to the boundary where arm `i` just becomes optimal, while staying inside the allowed model class for that arm. This gives the sharp information constant:

```text
liminf E[T_i(n)] / log n >= 1 / d_inf(P_i, mu*, M_i),
```

or, in a one-parameter family,

```text
liminf R_n / log n >= sum_{i: Delta_i > 0} Delta_i / KL(theta_i, theta*).
```

For Bernoulli arms the denominator is `kl(mu_i, mu*)`. For Gaussian rewards with known variance `sigma^2`, it is `Delta_i^2/(2 sigma^2)`, so the necessary number of pulls of arm `i` is `2 sigma^2 log n / Delta_i^2`. In the unit-variance case this is `2 log n / Delta_i^2`.

This is the main insight. Exploration is not an ad hoc tax I choose by taste. Every inferior arm has to be sampled until the policy has accumulated about `log n` units of evidence against the closest world where that arm is best. The denominator is not a decorative information-theory constant; it is the rate at which evidence arrives from samples of that arm.

Now I ask what kind of rule naturally pays that cost and stops. Greedy is too brittle because it pretends each empirical mean is already the truth. A barely sampled arm can be underestimated and then never repaired. Uniform random exploration is too blunt because it keeps paying for arms after the evidence is already decisive. I need a rule whose desire to sample an arm shrinks automatically as evidence accumulates.

The lower-bound picture suggests optimism. For each arm, I maintain the largest mean that is still plausible given its data, and I play the arm with the largest plausible mean. A lightly sampled arm receives a wide allowance because the data cannot yet rule out a favorable world. A heavily sampled bad arm receives a narrow allowance, so once its estimate settles below the best mean it stops attracting pulls. The same number is doing both jobs: it exploits high empirical means and explores high uncertainty.

For bounded rewards, Hoeffding's inequality tells me the size of the allowance. After `s` samples,

```text
P(hat_mu_s >= mu + a) <= exp(-2 s a^2),
P(hat_mu_s <= mu - a) <= exp(-2 s a^2).
```

Solving `exp(-2 s a^2) = delta` gives a one-sided width `sqrt(log(1/delta)/(2s))`. If I choose a time-dependent confidence level strong enough to survive a union bound over times and sample counts, I get the simple index

```text
hat_mu_i + sqrt(2 log t / T_i(t-1)).
```

The rule samples each arm once, then always selects the arm with the largest index.

The proof of its logarithmic regret has the same structure as the intuition. Fix a bad arm `i`. If arm `i` is chosen after it has already been sampled at least `ell` times, its index must beat the optimal arm's index. That can only happen if one of three things is true: the optimal arm's empirical mean is unusually low, arm `i`'s empirical mean is unusually high, or the confidence radius for arm `i` is still so large that the gap has not been separated. Hoeffding makes the first two events summable. The third is deterministic; it becomes false once

```text
T_i >= 8 log n / Delta_i^2.
```

Therefore

```text
E[T_i(n)] <= 8 log n / Delta_i^2 + 1 + pi^2/3.
```

Multiplying by `Delta_i` gives logarithmic regret. The constant is not the information constant in general. The proof spends slack in the quadratic Hoeffding radius and the union bound. But the order is right, and the mechanism matches the lower-bound obstruction: keep an arm alive exactly while the data still permit it to be the best.

This simple rule is not the original 1985 Lai-Robbins allocation rule; it is the later finite-time UCB1 index. It is still the clean constructive face of the same optimism idea, and it is faithful to the Auer-Cesa-Bianchi-Fischer pseudocode. To close the asymptotic constant for Gaussian rewards, I can tune the confidence schedule more carefully. In the 1-subgaussian normalization, a later anytime index with

```text
sqrt(2 log(1 + t log^2 t) / T_i(t-1))
```

has asymptotic regret at most

```text
sum_{i: Delta_i > 0} 2 log n / Delta_i,
```

which matches the lower bound for unit-variance Gaussians because `D(N(mu_i,1) || N(mu*,1)) = Delta_i^2/2`. In Bernoulli and exponential-family settings, the same lesson points to KL-shaped confidence sets rather than the quadratic Hoeffding radius: choose the largest mean whose KL divergence from the empirical model is still within the exploration budget.

So the discovery is not "put confidence intervals on a bandit." The deeper step is first proving that every good policy is secretly running a sequential test against neighboring worlds, with an unavoidable evidence bill of `log n / KL`. The confidence index is then the constructive face of that proof: it is an implementable way to keep sampling exactly those arms for which the favorable neighboring world has not yet been statistically ruled out.
