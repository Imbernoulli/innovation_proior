A decision maker faces $K$ slot machines, each arm $i$ returning i.i.d. rewards in $[0,1]$ from an unknown distribution with unknown mean $\mu_i$. Each round it picks one arm, sees only that arm's reward, and never learns the true means or the rewards of the arms it passed over. The honest measure of how well it does is regret, the expected shortfall against an oracle that always plays the best arm. Writing $\mu^* = \max_i \mu_i$, $\Delta_i = \mu^* - \mu_i$, and $T_i(n)$ for the number of times arm $i$ is pulled in the first $n$ rounds, the pull counts sum to $n$, so $\mu^* n = \sum_j \mu^* \, \mathbb{E}[T_j(n)]$ and the regret collapses to

$$R_n = \mu^* n - \sum_j \mu_j\, \mathbb{E}[T_j(n)] = \sum_{i:\,\mu_i < \mu^*} \Delta_i\, \mathbb{E}[T_i(n)].$$

The gaps $\Delta_i$ are fixed by the world; the only thing a policy controls is $\mathbb{E}[T_i(n)]$, how often it pulls each bad arm. So the whole game is to pull the suboptimal arms as rarely as possible — and the dilemma is that to know an arm is bad I must sample it, and every such sample is regret.

The obvious rule — estimate each mean by its empirical average $\bar x_j$ and always play the largest — can fail catastrophically. If the genuinely best arm returns a few unlucky low rewards early, its empirical mean drops below a mediocre arm's, the greedy rule stops choosing it, and because it is never pulled again its estimate never recovers: the policy locks onto a worse arm and pays gap-sized regret on every remaining round, $\Theta(n)$, linear. So some willingness to revisit under-sampled arms is not optional. The crude patch, $\varepsilon$-greedy — play the empirical best with probability $1-\varepsilon$ and a uniformly random arm with probability $\varepsilon$ — does break the lock-in, but a constant $\varepsilon$ keeps a constant fraction of all rounds random forever, and a random pull lands on a bad arm with constant probability, so it still pays $\Theta(n)$. Making $\varepsilon$ decay with time could in principle give sublinear regret, but tuning the decay correctly requires a lower bound on the smallest gap $\Delta$, which I do not know in advance. The deeper flaw is that this exploration is undirected: it spends pulls uniformly, blind to which arms I am actually unsure about, wasting them on arms I already know are bad. The prior upper-confidence-index rules of Lai and Robbins (1985) reached the logarithmic regret floor asymptotically but used an index that is generally hard to compute and depends on the whole reward history, tied to single-parameter families; Agrawal (1995) made the index a cheap function of the rewards seen so far but the guarantees stayed asymptotic. What is missing is a rule that is cheap to run, assumes only bounded rewards, and comes with a regret bound that holds at every finite horizon $n$, written explicitly in the gaps.

I propose UCB1. The idea is to replace each arm's empirical mean with an optimistic estimate — the highest value its true mean could plausibly take given the data — and play the largest. After playing each arm once to seed every count, at each round play

$$\arg\max_j \;\Big[\, \bar x_j + \sqrt{\tfrac{2 \ln n}{N_j}} \,\Big],$$

where $\bar x_j$ is arm $j$'s current empirical mean, $N_j$ its pull count, and $n$ the total plays so far. The first term exploits; the bonus $\sqrt{2\ln n / N_j}$ is the half-width of a one-sided confidence interval and is what makes the exploration targeted. A well-sampled good arm has a tight bonus and a high optimistic value, so it is exploited; a well-sampled bad arm has a tight bonus and a low value, so it is correctly left alone; an under-sampled arm has a loose bonus that inflates its optimistic value well above its empirical mean, which pulls me toward sampling it. Crucially, misplaced optimism self-extinguishes: pulling a wrongly-inflated arm tightens its interval, and its optimistic value collapses toward the true low mean, so the policy stops on its own. That is exactly the property $\varepsilon$-greedy lacked — exploration directed by uncertainty and self-correcting when the uncertainty resolves the wrong way.

The bonus comes from concentration. The Chernoff-Hoeffding inequality says that for $s$ samples in $[0,1]$ with conditional mean $\mu$, $\mathbb{P}\{\bar x \ge \mu + a\} \le e^{-2 s a^2}$ and symmetrically below. To make a one-sided confidence statement fail with probability at most $\delta$, set $e^{-2 s a^2} = \delta$, giving radius $a = \sqrt{\ln(1/\delta)/(2s)}$. I tie the failure level to the clock by choosing $\delta = t^{-4}$, so $\ln(1/\delta) = 4\ln t$ and the confidence radius is $c_{t,s} = \sqrt{2 \ln t / s}$. This already has both behaviors I wanted: it shrinks like $1/\sqrt{s}$ as an arm is pulled, descending toward the truth so good arms are exploited and bad arms abandoned, and it grows like $\sqrt{\ln t}$ over time for arms left unpulled, so no arm is dismissed forever. The exponent $4$ is not free; it is the smallest clean choice that makes the regret proof's union bound converge, which is what the derivation forces.

To see that, I bound the pulls of a fixed suboptimal arm $i$. Arm $i$ is chosen at round $t$ only if its index beats the optimal arm's, so $T_i(n) = 1 + \sum_{t} \mathbf{1}\{I_t = i\}$, and conceding the first $\ell$ pulls and union-bounding over all possible sample counts $s$ for the optimal arm and $s_i \ge \ell$ for arm $i$,

$$T_i(n) \le \ell + \sum_{t=1}^{\infty}\sum_{s=1}^{t-1}\sum_{s_i=\ell}^{t-1} \mathbf{1}\{\, \bar x^*_s + c_{t,s} \le \bar x_{i,s_i} + c_{t,s_i} \,\}.$$

The event that the optimal arm's optimistic value comes out no larger than $i$'s implies at least one of an exhaustive trichotomy: (7) $\bar x^*_s \le \mu^* - c_{t,s}$, the optimal arm under-estimated by a full radius; (8) $\bar x_{i,s_i} \ge \mu_i + c_{t,s_i}$, arm $i$ over-estimated by a full radius; or (9) $\mu^* < \mu_i + 2 c_{t,s_i}$, the radius for $i$ so wide that even with no estimation error its optimistic value could top $\mu^*$. If none of these holds, then $\bar x^*_s + c_{t,s} > \mu^* \ge \mu_i + 2c_{t,s_i} > \bar x_{i,s_i} + c_{t,s_i}$, the strict negation of the event, so the trichotomy is genuine. Now I choose $\ell$ to kill (9) outright. Event (9) is $\Delta_i < 2\sqrt{2\ln t / s_i}$, a purely deterministic statement; squaring, it fails once $s_i \ge 8 \ln t / \Delta_i^2$, and since I only sum $t \le n$, taking $\ell = \lceil 8 \ln n / \Delta_i^2 \rceil$ makes (9) false throughout. That is where the constant $8$ comes from, and it is forced, not tuned: the radius carries the Hoeffding $2$, I need twice the radius to be at most $\Delta_i$ (one radius of slack for each arm), and $(2\sqrt 2)^2 = 8$. With (9) gone, only the probabilistic events remain, and Hoeffding bounds them exactly: $\mathbb{P}(7) \le e^{-2 s c_{t,s}^2} = e^{-4\ln t} = t^{-4}$, and likewise $\mathbb{P}(8) \le t^{-4}$ — this is the payoff of the $t^{-4}$ choice, the Hoeffding $2$ times the radius's $2$ giving the exponent $4$. Taking expectations, each round contributes at most $\sim t^2$ index pairs each of probability $\le 2t^{-4}$, hence $\le 2t^{-2}$ per round, and

$$\mathbb{E}[T_i(n)] \le \frac{8 \ln n}{\Delta_i^2} + 1 + \sum_{t \ge 1} \frac{2}{t^2} = \frac{8 \ln n}{\Delta_i^2} + 1 + \frac{\pi^2}{3},$$

using $\lceil x \rceil \le x+1$ and the Basel sum $\sum_t 1/t^2 = \pi^2/6$. Here is exactly why $\delta = t^{-4}$ and nothing milder: a softer $\delta = t^{-2}$ would leave $t^2 \cdot t^{-2} = 1$ per round, summing to infinity and collapsing the bound; the exponent $4$ is the smallest one whose post-union-bound series converges. Multiplying by $\Delta_i$ and summing over suboptimal arms gives the finite-time regret bound,

$$R_n \le 8 \sum_{i:\,\mu_i < \mu^*} \frac{\ln n}{\Delta_i} + \Big(1 + \frac{\pi^2}{3}\Big)\sum_{j=1}^K \Delta_j,$$

logarithmic in $n$ and valid at every finite horizon — exactly the hole the prior art left. Read without committing to a gap profile, split the arms at a threshold $\gamma$: arms with $\Delta_i \le \gamma$ cost at most $\gamma$ per pull over only $n$ total pulls, contributing $\le n\gamma$, while arms with $\Delta_i > \gamma$ contribute $\le 8K\ln n/\gamma$ in the leading term; balancing at $\gamma = \sqrt{8K\ln n / n}$ gives the gap-free reading $R_n = O(\sqrt{K n \ln n})$, so $R_n / n \to 0$.

A few points worth keeping in view. Against the Lai-Robbins floor of $(\ln n)/D(p_i\|p^*)$ pulls, where for bounded rewards $D(p_i\|p^*) \ge 2\Delta_i^2$ with constant $2$ best possible, UCB1's $8\ln n/\Delta_i^2$ has the right order but a leading constant about a factor $16$ above the floor; that looseness traces straight to the $8$, which an epoch-batched radius (play arms in geometrically growing batches of length $\tau(r) = \lceil (1+\alpha)^r \rceil$ with radius $\sqrt{(1+\alpha)\ln(en/\tau(r))/(2\tau(r))}$) can shave toward $1/(2\Delta_i^2)$ as $\alpha \to 0$, trading against a constant term that grows, so $\alpha$ is decayed slowly with $n$. Hoeffding needs only $\mathbb{E}[X_{i,t}\mid X_{i,1\ldots t-1}] = \mu_i$ and range $[0,1]$, so the bound survives mild dependence and non-i.i.d. rewards within an arm. And when the noise scale is the variance rather than a known range — Gaussian arms with unknown mean and variance — the same optimism template uses a sample-variance radius, the index $\bar x_j + \sqrt{16\,(q_j - N_j \bar x_j^2)/(N_j-1)\cdot \ln(n-1)/N_j}$ with $q_j$ the sum of squared rewards, paired with playing any arm pulled fewer than $\lceil 8\log n\rceil$ times; under numerically checked chi-squared and Student tail bounds this gives regret $\le 256(\log n)\sum_i \sigma_i^2/\Delta_i + (1 + \pi^2/2 + 8\log n)\sum_j \Delta_j$, the same order with the per-arm cost scaled by $\sigma_i^2$.

The implementation is just running sufficient statistics and an argmax: a base class holding the clock $t$, per-arm pull counts, and normalized cumulative rewards; a generic index policy that scores every arm and plays a maximizer with random tie-breaking; unpulled arms return $+\infty$, which forces the initial one-pull-per-arm phase; and the UCB index $\bar x_j + \sqrt{2\log t / N_j}$, recovered by $\texttt{UCBalpha}(\alpha=4)$.

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
