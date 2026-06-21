The stochastic multi-armed bandit problem starts with K independent reward sources, each with an unknown fixed distribution and unknown mean mu_i. At every round you choose one source, observe only its reward, and move on. The natural objective is regret, defined as R_n = n mu* - E[sum of rewards], where mu* is the largest mean. This regret can be rewritten as a weighted sum of the expected pulls of every suboptimal arm, so the whole problem reduces to controlling how many times each inferior source is sampled.

Several earlier ideas address pieces of the problem but leave important gaps. A greedy rule that always picks the source with the highest current sample mean is simple, but it can lock in early: the true best arm may look worse than it is after a few unlucky samples and then never recover. Forcing a constant fraction of random exploration avoids lock-in, yet it keeps paying for bad arms forever and cannot achieve vanishing average regret. Thompson sampling is principled, but it is Bayesian and does not immediately give a frequentist regret rate or a comparison against an information-theoretic lower bound. Robbins's play-the-winner rule shows that sequential design can outperform fixed allocation, yet it remains a heuristic without a characterization of the best possible rate. What is missing is both a proof of the unavoidable exploration cost and a single rule that pays that cost automatically.

The new method is Upper Confidence Bound, specifically UCB1. It is an optimism-based index policy introduced by Auer, Cesa-Bianchi, and Fischer for finite-time analysis. The idea is to maintain, for each arm, an upper confidence bound on its unknown mean rather than trusting the empirical mean alone. Arms with few samples receive a wide bonus that encourages exploration, while arms with many samples have a narrow bonus and are explored only when the evidence still permits them to be best.

The policy works as follows. After pulling every arm once, at round t it selects the arm maximizing hat_mu_i + sqrt(2 log t / T_i(t-1)), where hat_mu_i is the current empirical mean of arm i and T_i(t-1) is the number of times it has been pulled before round t. The bonus is derived from the Chernoff-Hoeffding bound: it is the one-sided confidence radius that would make a deviation unlikely after T_i samples. Because the bonus decreases with the sample count, a bad arm eventually stops being chosen once its empirical mean has settled below the best mean and its remaining uncertainty is too small to close the gap.

This construction directly reflects the Lai-Robbins lower bound. That bound says any policy that is consistent across the bandit class must pull each suboptimal arm i at least on the order of log n divided by the Kullback-Leibler divergence to the closest alternative world where i is best. UCB1 does not always match the exact information constant for non-Gaussian reward families, but it achieves the optimal logarithmic regret order for bounded rewards. In the refined 1-subgaussian version, using the index sqrt(2 log(1 + t log^2 t) / T_i(t-1)) gives a regret upper bound whose leading constant matches the Lai-Robbins lower bound for unit-variance Gaussian rewards. For Bernoulli or general exponential families, the same principle is captured even more tightly by KL-UCB, which replaces the quadratic bonus with a KL-based confidence set.

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

def select_ucb1(stats, t):
    """Auer-Cesa-Bianchi-Fischer UCB1 selector."""
    unplayed = np.flatnonzero(stats.counts == 0)
    if unplayed.size:
        return int(unplayed[0])
    index = stats.means + np.sqrt(2.0 * np.log(t) / stats.counts)
    return int(np.argmax(index))

def select_asymptotic_ucb(stats, t):
    """Refined 1-subgaussian UCB with f(t) = 1 + t log^2 t."""
    unplayed = np.flatnonzero(stats.counts == 0)
    if unplayed.size:
        return int(unplayed[0])
    log_t = np.log(max(t, 2))
    f_t = 1.0 + t * log_t * log_t
    index = stats.means + np.sqrt(2.0 * np.log(f_t) / stats.counts)
    return int(np.argmax(index))

# Example run with UCB1 on a small Bernoulli bandit.
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    means = [0.2, 0.5, 0.75]
    env = BanditEnv(means, rng)
    stats = ArmStats(len(means))
    n_rounds = 10000
    rewards = []

    for t in range(1, n_rounds + 1):
        arm = select_ucb1(stats, t)
        reward = env.pull(arm)
        stats.update(arm, reward)
        rewards.append(reward)

    cumulative_regret = n_rounds * env.mu_star - sum(rewards)
    print(f"empirical regret after {n_rounds} rounds: {cumulative_regret:.2f}")
    print("pull counts:", stats.counts)
    print("estimated means:", stats.means)
```
