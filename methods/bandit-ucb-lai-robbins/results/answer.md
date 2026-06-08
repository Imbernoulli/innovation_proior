# UCB and the Lai–Robbins lower bound

## Problem

A $K$-armed stochastic bandit: arm $i$ returns rewards i.i.d. from an unknown distribution $P_i$ with mean $\mu_i$; $\mu^\* = \max_i \mu_i$, gap $\Delta_i = \mu^\*-\mu_i$. Over $n$ pulls, minimize the regret

$$ R_n = n\mu^\* - \mathbb{E}\Big[\sum_t X_{I_t,t}\Big] = \sum_{i:\Delta_i>0}\Delta_i\,\mathbb{E}[T_i(n)], $$

where $T_i(n)$ is the number of pulls of arm $i$. The whole problem reduces to controlling the suboptimal pull counts $\mathbb{E}[T_i(n)]$.

## Key idea

Two matched results pin the answer.

- **A floor of $\Theta(\log n)$ with a KL constant.** No "consistent" policy (one with $R_n=o(n^p)$ for every $p>0$) can beat logarithmic regret, and the leading constant is governed by the Kullback–Leibler divergence to the optimal arm. This is the **Lai–Robbins lower bound**, proved by a change-of-measure argument: a bad arm must be pulled enough to statistically distinguish the instance from a neighboring one in which that arm is secretly best.
- **Optimism attains the logarithmic order.** The **Upper Confidence Bound (UCB)** rule assigns each arm an optimistic value — its empirical mean plus a confidence width — and plays the largest. Hoeffding's inequality fixes the width; the union bound fixes the confidence schedule; the per-arm pull count is then $O(\log n/\Delta_i^2)$. In the unit-variance Gaussian case, a tuned 1-subgaussian variant reaches the exact $2/\Delta_i$ regret constant.

## The Lai–Robbins lower bound

For any consistent policy, on any instance, every suboptimal arm satisfies

$$ \liminf_{n\to\infty}\frac{\mathbb{E}[T_i(n)]}{\log n} \ge \frac{1}{D(P_i\Vert P^\*)}, \qquad\text{hence}\qquad \liminf_{n\to\infty}\frac{R_n}{\log n} \ge \sum_{i:\Delta_i>0}\frac{\Delta_i}{D(P_i\Vert P^\*)}. $$

**Proof (change of measure).** Fix suboptimal arm $i$. Build a twin instance $\nu'$ identical except $P_i\to P_i'$ with mean above $\mu^\*$ (so arm $i$ is optimal in $\nu'$) and $D(P_i\Vert P_i')$ within $\varepsilon$ of $\inf\{D(P_i\Vert P'):\mu(P')>\mu^\*\}$. Two routes give the same constant:

*Likelihood-ratio route.* With $\widehat{\mathrm{kl}}_s = \sum_{t\le s}\log\frac{p_i(X_{i,t})}{p_i'(X_{i,t})}$ (mean $D(P_i\Vert P_i')$ under $P_i$), the full-history change-of-measure identity is
$$ \mathbb{P}_\nu(A)=\mathbb{E}_{\nu'}[\mathbf 1_A e^{\widehat{\mathrm{kl}}_{T_i(n)}}]. $$
Let $f_n=(1-\varepsilon)\log n/D(P_i\Vert P_i')$ and $C_n=\{T_i(n)<f_n,\ \widehat{\mathrm{kl}}_{T_i(n)}\le(1-\varepsilon/2)\log n\}$. On $C_n$, $e^{\widehat{\mathrm{kl}}}\le n^{1-\varepsilon/2}$, so $\mathbb{P}_\nu(C_n)\le n^{1-\varepsilon/2}\mathbb{P}_{\nu'}(C_n)$. Consistency in $\nu'$ gives $\mathbb{E}_{\nu'}[n-T_i(n)]=o(n^a)$ for every $a>0$, hence $\mathbb{P}_{\nu'}(C_n)=o(n^{a-1})$; choosing $a<\varepsilon/2$ gives $\mathbb{P}_\nu(C_n)\to0$. The maximal SLLN makes the likelihood-ratio clause typical on $\{T_i(n)<f_n\}$, so $\mathbb{P}_\nu(T_i(n)<f_n)\to0$ and $\mathbb{E}_\nu[T_i(n)]\ge(1-o(1))f_n$.

*Information-inequality route.* The chain rule gives
$$ D(\mathbb{P}_\nu\Vert\mathbb{P}_{\nu'}) = \sum_j \mathbb{E}_\nu[T_j(n)]\,D(P_j\Vert P_j') = \mathbb{E}_\nu[T_i(n)]\,D(P_i\Vert P_i'), $$
because the policy action kernel is the same in both worlds and only arm $i$ differs. Bretagnolle–Huber gives $\mathbb{P}_\nu(A)+\mathbb{P}_{\nu'}(A^c)\ge\frac12 e^{-D(\mathbb{P}_\nu\Vert\mathbb{P}_{\nu'})}$. With $A=\{T_i(n)>n/2\}$ and $\gamma=\min\{\Delta_i,\mu_i'-\mu^\*\}$,
$$ R_n+R_n' \ge \frac{n\gamma}{4}\exp\{-\mathbb{E}_\nu[T_i(n)]D(P_i\Vert P_i')\}. $$
Since consistency makes $R_n+R_n'=o(n^p)$ for every $p>0$, rearranging gives $\mathbb{E}_\nu[T_i(n)]D(P_i\Vert P_i')\ge(1-p)\log n+O(1)$, then $p\downarrow0$.

Send $P_i'$ to the boundary ($D(P_i\Vert P_i')\to D(P_i\Vert P^\*)$) and $\varepsilon\to0$. Bernoulli: constant $1/d(\mu_i,\mu^\*)$. Unit-variance Gaussian: $D=\Delta_i^2/2$, constant $2/\Delta_i^2$.

## UCB and its finite-time upper bound

**Index.** After playing each arm once, at round $t$ play

$$ I_t = \arg\max_i\ \bar X_{i,T_i(t-1)} + \sqrt{\frac{2\log t}{T_i(t-1)}}. $$

The width $\sqrt{2\log t/s}$ is the inverted Hoeffding tail: $\mathbb{P}(\mu_i\ge\bar X_{i,s}+a)\le e^{-2sa^2}$, set to $\delta=t^{-4}$.

**Theorem (finite-time, $[0,1]$ rewards).** For every suboptimal arm,

$$ \mathbb{E}[T_i(n)] \le \frac{8\log n}{\Delta_i^2} + 1 + \frac{\pi^2}{3}, \qquad R_n \le \sum_{i:\Delta_i>0}\frac{8\log n}{\Delta_i} + \Big(1+\frac{\pi^2}{3}\Big)\sum_{i=1}^K\Delta_i. $$

**Proof.** With $c_{t,s}=\sqrt{2\log t/s}$ and threshold $\ell=\lceil 8\log n/\Delta_i^2\rceil$,
$$ T_i(n)\le \ell + \sum_{t}\sum_{s=1}^{t-1}\sum_{s_i=\ell}^{t-1}\mathbf 1\{\bar X^\*_s + c_{t,s}\le \bar X_{i,s_i}+c_{t,s_i}\}. $$
That inequality forces at least one of (7) $\bar X^\*_s\le\mu^\*-c_{t,s}$, (8) $\bar X_{i,s_i}\ge\mu_i+c_{t,s_i}$, (9) $\mu^\*<\mu_i+2c_{t,s_i}$. Hoeffding gives $\mathbb{P}(7),\mathbb{P}(8)\le e^{-4\log t}=t^{-4}$; (9) is impossible once $s_i\ge 8\log n/\Delta_i^2$ (since $\Delta_i-2\sqrt{2\log t/s_i}\ge 0$ for $t\le n$). Hence
$$ \mathbb{E}[T_i(n)]\le \frac{8\log n}{\Delta_i^2}+1+\sum_t\sum_{s}\sum_{s_i}2t^{-4}\le \frac{8\log n}{\Delta_i^2}+1+2\sum_t t^{-2}= \frac{8\log n}{\Delta_i^2}+1+\frac{\pi^2}{3}. $$

**Matching the constant.** For bounded rewards, Pinsker gives the quadratic scale $D(P_i\Vert P^\*)\ge 2\Delta_i^2$, so UCB1 has the right $\log n/\Delta_i^2$ order but not the sharp KL constant. In the 1-subgaussian/Gaussian normalization, tightening the confidence to $\delta(t)=1/f(t)$ with $f(t)=1+t\log^2 t$ and splitting at $\mu^\*-\varepsilon$ gives
$$ \mathbb{E}[T_i(n)]\le 1+\frac{5}{\varepsilon^2}+\frac{2(\log f(n)+\sqrt{\pi\log f(n)}+1)}{(\Delta_i-\varepsilon)^2}, \qquad \limsup_{n}\frac{R_n}{\log n}\le \sum_{i:\Delta_i>0}\frac{2}{\Delta_i}, $$
which for unit-variance Gaussian arms equals the Lai–Robbins floor $\sum_i\Delta_i/D(P_i\Vert P^\*)$ exactly.

## Implementation

```python
import numpy as np

class BanditEnv:
    """K arms; pulling arm i returns an i.i.d. reward with unknown mean mu[i]."""
    def __init__(self, means, rng):
        self.means = np.asarray(means, float)
        self.rng = rng
        self.mu_star = self.means.max()

    def pull(self, i):
        return float(self.rng.random() < self.means[i])

class ArmStats:
    """Running sufficient statistics per arm: pull count and empirical mean."""
    def __init__(self, K):
        self.counts = np.zeros(K, int)
        self.means  = np.zeros(K, float)

    def update(self, i, reward):
        self.counts[i] += 1
        n = self.counts[i]
        self.means[i] += (reward - self.means[i]) / n

def select_arm(stats, t):
    """UCB1: empirical mean plus sqrt(2 ln t / T_i)."""
    unplayed = np.flatnonzero(stats.counts == 0)
    if unplayed.size:
        return int(unplayed[0])
    index = stats.means + np.sqrt(2.0 * np.log(t) / stats.counts)
    return int(np.argmax(index))

def select_arm_asymptotically_optimal(stats, t):
    """1-subgaussian/Gaussian variant with f(t)=1+t log^2(t)."""
    unplayed = np.flatnonzero(stats.counts == 0)
    if unplayed.size:
        return int(unplayed[0])
    f_t = 1.0 + t * np.log(t) ** 2
    index = stats.means + np.sqrt(2.0 * np.log(f_t) / stats.counts)
    return int(np.argmax(index))

def run(env, n, K, selector=select_arm):
    stats = ArmStats(K)
    regret = 0.0
    for t in range(1, n + 1):
        i = selector(stats, t)
        reward = env.pull(i)
        stats.update(i, reward)
        regret += env.mu_star - env.means[i]
    return regret
```
