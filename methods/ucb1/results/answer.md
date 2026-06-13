# UCB1, distilled

UCB1 is a multi-armed bandit allocation rule that plays, each round, the arm with the highest
**upper confidence bound** on its mean: empirical mean plus a confidence radius derived from
the Chernoff-Hoeffding inequality. It implements optimism in the face of uncertainty — inflate
under-sampled arms so they get explored, let the inflation collapse once an arm is sampled
enough — and it is the first such rule with a **finite-time** logarithmic regret bound that
holds for every horizon, for arbitrary reward distributions bounded in `[0,1]`, while being
cheap to compute (a few running sums and an argmax per round).

## Problem it solves

`K`-armed stochastic bandit: arm `i` returns i.i.d. rewards in `[0,1]` with unknown mean
`mu_i`; only the pulled arm's reward is observed. With `mu* = max_i mu_i`,
`Delta_i = mu* - mu_i`, and `T_i(n)` the pulls of arm `i` in `n` rounds, the regret is

```
R_n = mu* n - sum_j mu_j E[T_j(n)] = sum_{i: mu_i < mu*} Delta_i E[T_i(n)].
```

Minimizing regret reduces to keeping each suboptimal arm's expected pull count `E[T_i(n)]`
small, against the exploration-exploitation tension: explore too little and an unlucky early
sample can permanently bury the best arm (linear regret); explore too much (or blindly, as a
constant-rate `eps`-greedy does) and you waste pulls on known-bad arms (also linear).

## Key idea

Replace each arm's empirical mean by an **optimistic** estimate — an upper confidence bound on
its true mean — and play the largest. After playing each arm once, at each round play

```
arg max_j  [ x_bar_j + sqrt( 2 ln n / N_j ) ],
```

where `x_bar_j` is arm `j`'s empirical mean, `N_j` its pull count, and `n` the total plays so
far. The bonus `sqrt(2 ln n / N_j)` is the half-width of a one-sided confidence interval:

- It **shrinks** like `1/sqrt(N_j)` as an arm is pulled, so a well-sampled arm's optimistic
  value descends toward its truth — good arms are exploited, bad arms abandoned.
- It **grows** like `sqrt(ln n)` over time for arms left unpulled, so no arm is dismissed
  forever — exploration is directed at whichever arm is currently most uncertain.
- Misplaced optimism self-corrects: pulling a wrongly-optimistic arm tightens its interval and
  its inflated value collapses, so the policy stops on its own.

## Deriving the radius (Chernoff-Hoeffding)

For `s` samples in `[0,1]` with mean `mu`, `P{ x_bar >= mu + a } <= e^{-2 s a^2}` (and
symmetrically below). To make a one-sided confidence statement fail with probability at most
`delta`, set `e^{-2 s a^2} = delta`, giving radius `a = sqrt( ln(1/delta) / (2 s) )`. Tying the
failure level to the clock, `delta = t^{-4}` (so `ln(1/delta) = 4 ln t`), yields the
confidence radius `c_{t,s} = sqrt( 2 ln t / s )`. The exponent `4` is the smallest clean choice
that makes the regret proof's union bound converge (below).

## Finite-time regret bound (Theorem)

For `K > 1`, arbitrary reward distributions with support in `[0,1]`, and any `n`:

```
E[T_i(n)] <= (8 / Delta_i^2) ln n + 1 + pi^2/3      for each suboptimal arm i,

R_n <= 8 sum_{i: mu_i < mu*} (ln n / Delta_i) + (1 + pi^2/3) sum_{j=1}^K Delta_j.
```

Logarithmic in `n` and valid at every finite horizon. Read without reference to a specific gap
profile, split arms at a threshold `gamma`: gaps `<= gamma` contribute at most `n gamma` total,
while gaps `> gamma` contribute at most `8 K ln n / gamma` in the leading term. Choosing
`gamma = sqrt(8 K ln n / n)` gives the gap-free reading `R_n = O(sqrt(K n ln n))`, so average
regret `R_n / n -> 0`.

### Proof sketch (per suboptimal arm `i`)

Let `c_{t,s} = sqrt(2 ln t / s)`, `ell = ceil(8 ln n / Delta_i^2)`. Counting only pulls after
`i` has been pulled `>= ell` times and union-bounding over all sample counts,

```
T_i(n) <= ell + sum_{t=1}^{inf} sum_{s=1}^{t-1} sum_{s_i=ell}^{t-1} 1{ x_bar*_s + c_{t,s} <= x_bar_{i,s_i} + c_{t,s_i} }.
```

The event implies at least one of: (7) `x_bar*_s <= mu* - c_{t,s}` (optimal arm
under-estimated), (8) `x_bar_{i,s_i} >= mu_i + c_{t,s_i}` (arm `i` over-estimated), (9)
`mu* < mu_i + 2 c_{t,s_i}` (interval too wide). Hoeffding gives
`P(7), P(8) <= e^{-4 ln t} = t^{-4}`. Event (9) is `Delta_i < 2 sqrt(2 ln t / s_i)`, which is
**false** once `s_i >= 8 ln n / Delta_i^2` (since then `2 c_{t,s_i} <= Delta_i`, using
`ln t <= ln n`) — this forces the constant `8`. Summing the surviving `t^{-4}` events over the
`~ t^2` index pairs,

```
E[T_i(n)] <= 8 ln n / Delta_i^2 + 1 + sum_{t>=1} 2 t^{-2} = 8 ln n / Delta_i^2 + 1 + pi^2/3,
```

using `sum_{t>=1} 1/t^2 = pi^2/6`. Multiply by `Delta_i` and sum over suboptimal arms.
(Hoeffding needs only `E[X_{i,t} | X_{i,1..t-1}] = mu_i` and range `[0,1]`, so the bound also
survives mild dependence and non-i.i.d. rewards within an arm.)

## Relation to the floor and to siblings

- **Lai-Robbins floor.** The unavoidable rate is `(ln n)/D(p_i || p*)` pulls of a bad arm,
  with `D` the KL divergence to the optimal arm. Since `D(p_i || p*) >= 2 Delta_i^2` (constant
  `2` best possible) for bounded rewards, UCB1's `8 ln n / Delta_i^2` has the right `ln n`
  order but a leading constant a factor `~16` above the floor.
- **Epoch-batched radius (chasing the constant).** Play arms in geometrically growing batches
  of length `tau(r) = ceil((1+alpha)^r)` with radius
  `a_{n,r} = sqrt( (1+alpha) ln(e n / tau(r)) / (2 tau(r)) )`; this brings the leading constant
  arbitrarily close to `1/(2 Delta_i^2)` as `alpha -> 0` (with a constant term that grows as
  `alpha -> 0`, so `alpha` is decayed slowly with `n`).
- **`eps_n`-greedy.** With a decaying exploration rate `eps_n = min{1, cK/(d^2 n)}` it also
  achieves logarithmic regret — but only given a known lower bound `d` on the smallest gap.
- **Unknown-variance Gaussian arms.** When the noise scale is the variance, not a known range,
  size the radius by the sample variance: index
  `x_bar_j + sqrt( 16 (q_j - N_j x_bar_j^2)/(N_j - 1) * ln(n-1)/N_j )` (with `q_j` the sum of
  squared rewards), plus "play any arm pulled `< ceil(8 log n)` times." The radius now rests on
  numerically checked chi-squared / Student tail bounds; under those tail bounds, regret is
  `<= 256 (log n) sum_i sigma_i^2/Delta_i + (1 + pi^2/2 + 8 log n) sum_j Delta_j` — same
  `log n` order, per-arm cost scaled by `sigma_i^2`.

## Working code

The implementation pattern is an index policy: a base class holds `t`, `pulls`, and cumulative
normalized `rewards`; an index policy computes per-arm indices and chooses a maximizer with random
tie-breaking; unpulled arms return `+inf`, which forces the initial one-pull-per-arm phase.

```python
from math import log, sqrt
import numpy as np


class BasePolicy:
    """Policy state: time, per-arm pull counts, and normalized cumulative rewards."""

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
    """Generic optimistic-index policy."""

    def __init__(self, nbArms, lower=0.0, amplitude=1.0):
        super().__init__(nbArms, lower=lower, amplitude=amplitude)
        self.index = np.zeros(nbArms, dtype=float)

    def computeIndex(self, arm):
        raise NotImplementedError

    def computeAllIndex(self):
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)

    def choice(self):
        self.computeAllIndex()
        best = np.flatnonzero(self.index == np.max(self.index))
        return int(np.random.choice(best))


class UCB(IndexPolicy):
    """Bounded-reward UCB index: x_bar_j + sqrt(2 log(t) / N_j)."""

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float("+inf")
        mean = self.rewards[arm] / self.pulls[arm]
        bonus = sqrt(2.0 * log(self.t) / self.pulls[arm])
        return mean + bonus

    def computeAllIndex(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            indexes = (self.rewards / self.pulls) + np.sqrt(
                2.0 * np.log(self.t) / self.pulls
            )
        indexes[self.pulls < 1] = float("+inf")
        self.index[:] = indexes


class UCBalpha(UCB):
    """Generalized index: x_bar_j + sqrt(alpha log(t) / (2 N_j)); alpha=4 is UCB1."""

    def __init__(self, nbArms, alpha=4, lower=0.0, amplitude=1.0):
        super().__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha >= 0
        self.alpha = alpha

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float("+inf")
        mean = self.rewards[arm] / self.pulls[arm]
        bonus = sqrt(self.alpha * log(self.t) / (2.0 * self.pulls[arm]))
        return mean + bonus

    def computeAllIndex(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            indexes = (self.rewards / self.pulls) + np.sqrt(
                self.alpha * np.log(self.t) / (2.0 * self.pulls)
            )
        indexes[self.pulls < 1] = float("+inf")
        self.index[:] = indexes
```

`UCBalpha(alpha=4)` recovers UCB1's `sqrt(2 ln t / N_j)` bonus.
