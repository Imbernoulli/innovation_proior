## Decision Problem

There are `K` independent reward sources. Source `i` produces observations with a fixed but unknown distribution `P_i` and unknown mean `mu_i`. At each round I choose one source, see only that reward, and then choose again using the accumulated history. The benchmark is the clairvoyant rule that always uses a source with mean `mu* = max_i mu_i`.

The natural loss is regret,

```text
R_n = n mu* - E[sum_{t=1}^n X_{I_t,t}]
    = sum_{i: mu_i < mu*} (mu* - mu_i) E[T_i(n)].
```

Here `T_i(n)` is the number of pulls of source `i` in the first `n` rounds. This identity expresses regret through the expected number of times each inferior source is sampled.

## Available Statistical Language

Several tools are already on the table. Empirical means concentrate as sample size grows. For bounded rewards in `[0,1]`, if `hat_mu_s` is the average of `s` independent observations with mean `mu`, the Chernoff-Hoeffding bound gives tails of the form

```text
P(hat_mu_s >= mu + a) <= exp(-2 s a^2),
P(hat_mu_s <= mu - a) <= exp(-2 s a^2).
```

There is also a testing language for comparing two possible reward laws. Likelihood ratios accumulate additively over independent samples, and relative entropy measures the expected per-sample log-likelihood growth under one law against another. These facts are standard in sequential testing.

## Earlier Allocation Rules

Thompson's treatment-allocation proposal is Bayesian. After observing two samples, compute the posterior probability that one treatment's success probability exceeds the other's, then allocate future subjects according to a monotone function of that probability. It avoids a hard premature commitment and is motivated by reducing expected sacrifice, and it depends on a prior.

Robbins's sequential-design formulation removes the terminal-decision framing. The problem is not to test equality or estimate a difference after data collection; the problem is how to draw the sample itself so that the expected cumulative reward is large. His play-the-winner rule for two coins is analyzable and can beat fixed allocation.

Greedy empirical choice is the obvious computational baseline: always use the source with the largest current sample mean. Constant random exploration mixes a fixed fraction of uniformly random draws into the schedule.

## Evaluation Setting

The evaluation is asymptotic and finite-time. Asymptotically, a rule is judged by the leading order of its average regret in `n`. Finite-time, a rule is judged by its explicit pull-count or regret bounds, stated in terms of the gaps `Delta_i = mu* - mu_i`.

The hard cases are close alternatives. If two reward laws have similar means or similar likelihoods, then the data needed to tell them apart may be large even when one is inferior. A rule is therefore judged both on whether it identifies the best source and on how much reward it accrues along the way.

## Implementation Frame

The environment, running means, and regret accounting are fixed. The open slot is the arm-selection rule.

```python
import numpy as np

class BanditEnv:
    def __init__(self, means, rng):
        self.means = np.asarray(means, float)
        self.rng = rng
        self.mu_star = self.means.max()

    def pull(self, i):
        return float(self.rng.random() < self.means[i])

class ArmStats:
    def __init__(self, K):
        self.counts = np.zeros(K, int)
        self.means = np.zeros(K, float)

    def update(self, i, reward):
        self.counts[i] += 1
        n = self.counts[i]
        self.means[i] += (reward - self.means[i]) / n

def select_arm(stats, t):
    # Decide from counts, empirical means, and the current round.
    pass
```
