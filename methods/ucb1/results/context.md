# Context: the stochastic multi-armed bandit and the cost of exploration (circa late 1990s)

## Research question

A decision maker faces `K` slot machines ("arms"). Pulling arm `i` returns a random reward
`X_{i,1}, X_{i,2}, ...` drawn i.i.d. from an unknown distribution `P_i` with unknown mean
`mu_i`, and the rewards are bounded — without loss of generality in `[0,1]`. At each round the
agent picks one arm based only on the arms it has already pulled and the rewards it has already
seen, observes that arm's reward, and moves on. It never sees the rewards of the arms it did
not pull, and it never learns the true means. The agent wants its total reward over `n` rounds
to be as large as possible.

Because the means are unknown, the honest yardstick is not absolute reward but **regret** — the
expected shortfall against an oracle that always plays the best arm. With `mu* = max_i mu_i`
and `T_i(n)` the number of times arm `i` is played in the first `n` rounds, the regret after
`n` plays is

```
R_n = mu* n - sum_{j=1}^K mu_j E[T_j(n)] = sum_{j: mu_j < mu*} Delta_j E[T_j(n)],
      Delta_j = mu* - mu_j.
```

So the regret is exactly the sum, over the suboptimal arms, of each arm's gap times how often
the policy pulls it. Every suboptimal pull costs its gap; the whole problem is to keep the
expected suboptimal pull counts `E[T_j(n)]` small.

The dilemma is structural. To find the best arm the agent must *explore* — sample arms whose
means it is still unsure of — but every exploratory pull of a truly worse arm is regret. Yet
without exploration a single unlucky early sample of the best arm could make it look bad and
bury it forever, costing regret that grows linearly in `n`. The goal is a policy whose regret
grows as slowly as possible.

## Background

The bandit problem is a long-studied model of sequential decision under uncertainty, central
in statistics (Berry & Fristedt, 1985) and, by this time, in reinforcement learning (Sutton &
Barto, 1998) and evolutionary programming (Holland, 1992). Three pieces of prior understanding
are load-bearing.

**Regret has a logarithmic floor.** Lai & Robbins (1985), for reward distributions indexed by
a single real parameter, established that no good allocation rule can do better than
logarithmic regret, and pinned the leading constant. For any allocation strategy and any
suboptimal arm `j`, asymptotically

```
E[T_j(n)] >= (ln n) / D(p_j || p*),
```

where `D(p_j || p*) = integral p_j ln(p_j / p*)` is the Kullback-Leibler divergence between the
reward density of a suboptimal arm and that of the best arm. The intuition is informational: to
be sure arm `j` is worse, the agent must collect enough samples to statistically distinguish
`P_j` from a hypothetical distribution under which `j` would be best, and KL divergence measures
exactly how many samples that takes. So `Theta(ln n)` pulls of each bad arm is unavoidable, and
the best one can hope for is to match that order with the right constant.

**Index rules reach the floor.** Lai & Robbins (1985) also exhibited rules attaining the bound,
via a quantity they attached to each arm and called an "upper confidence index": play the arm
whose index is largest, where the index is an optimistic estimate of the arm's mean. Their index
depends on the entire sequence of rewards seen from the arm, and the construction is tied to
single-parameter families. Agrawal (1995) introduced a family of index rules whose index is a
*simple function of the total reward obtained so far* from the arm, while keeping the optimal
logarithmic order.

**A concentration tool sets the size of optimism.** For bounded random variables the
Chernoff-Hoeffding inequality controls how far an empirical average can stray from its mean.
For `X_1, ..., X_n` with values in `[0,1]` and `E[X_t | X_1, ..., X_{t-1}] = mu`, writing
`S_n = X_1 + ... + X_n`, for all `a >= 0`

```
P{ S_n >= n mu + a } <= e^{-2 a^2 / n}      and      P{ S_n <= n mu - a } <= e^{-2 a^2 / n}.
```

Equivalently, for the empirical mean `X_bar = S_n / n` and any `eps >= 0`,
`P{ X_bar >= mu + eps } <= e^{-2 n eps^2}` and symmetrically below. The exponent's factor of 2
is the property that fixes how wide a confidence interval of a given confidence level must be.
(The companion Bernstein/empirical-variance inequality
`P{ S_n >= E[S_n] + a } <= exp(-(a^2/2)/(sigma^2 + a/2))` is the variance-aware refinement used
when the reward variance, rather than just the range, is the relevant scale.)

A policy that simply plays the arm with the highest *empirical* mean can get permanently stuck.
If the best arm yields a few unlucky low samples early, its empirical mean drops below a
mediocre arm's, the greedy rule stops pulling it, and because it is never pulled again its
estimate never recovers — the policy locks onto the wrong arm and pays gap-sized regret on
every remaining round, i.e. `Theta(n)` total.

## Baselines

**Greedy / "play the empirical best" (folklore; the eps-greedy fix, Sutton & Barto 1998).**
Track each arm's empirical mean and play the largest. The standard variant is `eps`-greedy:
with probability `1 - eps` play the empirical best, with probability `eps` play a uniformly
random arm.

**Lai & Robbins (1985) upper-confidence-index rules.** Attach to each arm an index — an
optimistic estimate of its mean built from that arm's reward history — and play the arm of
largest index. This achieves `E[T_j(n)] <= (1/D(p_j || p*) + o(1)) ln n`, matching the lower
bound's order and constant asymptotically.

**Agrawal (1995) simple-index rules.** A family of index policies whose index is a simple
function of the total reward seen so far from each arm, much easier to compute than Lai &
Robbins', retaining the optimal logarithmic order (with a possibly larger leading constant).

## Evaluation settings

The natural yardsticks for an allocation rule, all defined before any particular rule exists:

- **Bounded stochastic bandits.** `K` arms with i.i.d. rewards in `[0,1]` and fixed unknown
  means; horizon `n` rounds. The canonical case is Bernoulli arms (each pull returns `0`/`1`
  with arm-specific probability). Metric: cumulative regret `R_n = sum_j Delta_j E[T_j(n)]`,
  averaged over independent runs, read as a function of `n`. The gap profile `{Delta_j}` and
  the number of arms `K` are the levers; one reports both the gap-dependent regret and its
  gap-free (worst-case-over-gaps) reading.
- **Gaussian / unknown-variance arms.** Rewards normally distributed with unknown mean *and*
  unknown variance — the natural setting where the relevant scale of fluctuation is the
  variance, not a known bounded range.
- **Reward families used in the asymptotic theory.** Normal, Bernoulli, Poisson, exponential —
  the single-parameter families for which the `1/D(p_j||p*)` constant is concrete, used to
  compare a rule's leading constant against the Lai-Robbins floor.
- Protocol: fix the means, run the policy for `n` rounds, accumulate regret; repeat over many
  random seeds and report the mean regret curve against `n`, and the per-arm pull counts
  `E[T_j(n)]`.

## Code framework

An allocation policy plugs into a fixed bandit interaction loop: each round the harness asks
the policy which arm to pull, draws that arm's stochastic reward from the hidden environment,
and hands the reward back to the policy to update its state. A compact scaffold is an index
policy. The base class keeps only the global clock, per-arm pull counts, and per-arm cumulative
rewards; the index wrapper computes one score per arm and chooses a maximizer. What remains
empty is the score assigned to one arm from those running statistics.

```python
import numpy as np


class BasePolicy:
    """Policy state shared by allocation rules."""

    def __init__(self, nbArms, lower=0.0, amplitude=1.0):
        self.nbArms = nbArms
        self.lower = lower
        self.amplitude = amplitude
        self.t = 0
        self.pulls = np.zeros(nbArms, dtype=int)
        self.rewards = np.zeros(nbArms, dtype=float)

    def startGame(self):
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0.0)

    def getReward(self, arm, reward):
        self.t += 1
        self.pulls[arm] += 1
        self.rewards[arm] += (reward - self.lower) / self.amplitude


class IndexPolicy(BasePolicy):
    """Score each arm and choose an arm with maximal score."""

    def __init__(self, nbArms, lower=0.0, amplitude=1.0):
        super().__init__(nbArms, lower=lower, amplitude=amplitude)
        self.index = np.zeros(nbArms, dtype=float)

    def computeIndex(self, arm):
        # TODO: assign an arm score from t, pulls[arm], and rewards[arm].
        pass

    def computeAllIndex(self):
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)

    def choice(self):
        self.computeAllIndex()
        best = np.flatnonzero(self.index == np.max(self.index))
        return int(np.random.choice(best))


# the fixed interaction loop the policy plugs into
def run(policy, env, n):
    policy.startGame()
    for _ in range(n):
        arm = policy.choice()
        reward = env.pull(arm)
        policy.getReward(arm, reward)
```

The loop, statistics, and argmax wrapper are settled; `computeIndex` is where the allocation rule
will live.
