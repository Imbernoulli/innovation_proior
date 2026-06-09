# Context

## Research question

I am handed a gambling machine with $K$ arms. Pulling arm $i$ returns a random reward $X_{i,1}, X_{i,2}, \dots$, drawn i.i.d. from some fixed but unknown distribution $P_i$ with unknown mean $\mu_i$. The rewards across arms are independent. I get $n$ pulls, choosing which arm to pull at each step using only what I have seen so far, and I want the largest possible total reward $\sum_{t} X_{I_t, t}$.

If I knew the means, I would pull the best arm $i^\* = \arg\max_i \mu_i$ every time and collect $n\mu^\*$ where $\mu^\* = \max_i \mu_i$. Because I do not know them, I will sometimes pull worse arms. The natural cost is the **regret**

$$ R_n \;=\; n\mu^\* - \mathbb{E}\!\Big[\sum_{t=1}^{n} X_{I_t,t}\Big] \;=\; \sum_{i:\,\mu_i<\mu^\*} (\mu^\*-\mu_i)\,\mathbb{E}[T_i(n)], $$

where $T_i(n)$ is the number of times arm $i$ was pulled in the first $n$ rounds and $\Delta_i = \mu^\*-\mu_i$ is the suboptimality gap. The regret is just the gaps weighted by how often each bad arm was pulled.

The precise question is two-sided. (1) **How small can the regret possibly be?** Is there a fundamental floor — some growth rate in $n$ that no policy, however clever, can beat — and what controls its size? (2) **Is that floor achievable by an explicit, simple, computable rule?** A solution must resolve the exploration/exploitation tension: pull an arm enough to learn its mean, but not so much that the pulls of bad arms add up. The pain is that these two demands pull in opposite directions, and a rule that gets one right tends to get the other wrong.

## Background

The problem sits inside *sequential design of experiments* — the idea that the choice of what to sample next can and should depend on what has already been observed, rather than being fixed in advance. Robbins (1952) made this the central question: given two populations $A,B$ with unknown means $\alpha,\beta$, "how should we draw a sample $x_1,\dots,x_n$ from the two populations if our object is to achieve the greatest possible expected value of the sum $S_n = x_1+\cdots+x_n$?" He framed it as a simplified model of "the general question of how we learn — or should learn — from past experience," explicitly outside the testing/estimation paradigm: there is no terminal decision, the whole problem is *how to draw the sample*.

Several technical facts about the design space are settled and available before any optimal rule is in hand.

**Concentration of empirical means.** The empirical mean $\bar X_{i,s} = \frac1s\sum_{t=1}^s X_{i,t}$ of $s$ i.i.d. samples concentrates around $\mu_i$ at rate $1/\sqrt s$. For rewards bounded in $[0,1]$, the Chernoff–Hoeffding bound gives, for any $a\ge 0$,

$$ \mathbb{P}\{\bar X_{i,s} \ge \mu_i + a\} \le e^{-2sa^2}, \qquad \mathbb{P}\{\bar X_{i,s} \le \mu_i - a\} \le e^{-2sa^2}. $$

More abstractly, if $\ln \mathbb{E}\,e^{\lambda(X-\mu)} \le \psi(\lambda)$ for a convex $\psi$ (Hoeffding's lemma gives $\psi(\lambda)=\lambda^2/8$ on $[0,1]$), then $\mathbb{P}(\bar X_{i,s}-\mu_i > \varepsilon) \le e^{-s\psi^\*(\varepsilon)}$ with $\psi^\*$ the Legendre–Fenchel transform; for $[0,1]$, $\psi^\*(\varepsilon)=2\varepsilon^2$, recovering Hoeffding.

**Kullback–Leibler divergence and hypothesis testing.** For two distributions, $D(P\Vert Q) = \int p\log(p/q)$. For Bernoulli means $p,q$, $d(p,q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$; for two unit-variance Gaussians, $D(\mathcal N(\mu_1,1)\Vert\mathcal N(\mu_2,1)) = (\mu_1-\mu_2)^2/2$. Pinsker's inequality relates it to the gap: $2(p-q)^2 \le d(p,q)$, and a one-line expansion gives $d(p,q)\le (p-q)^2/(q(1-q))$. The divergence relates to the likelihood ratio: $\log \frac{dP}{dQ}$ accumulated over i.i.d. samples has expectation growing at rate $D(P\Vert Q)$ under $P$.

**The regret decomposition.** Because $R_n = \sum_i \Delta_i\,\mathbb{E}[T_i(n)]$, the entire problem reduces to controlling the expected pull counts $\mathbb{E}[T_i(n)]$ of suboptimal arms; the gaps are fixed by the instance. Any lower bound on regret is a lower bound on these counts, and any policy's regret guarantee is an upper bound on them.

The prevailing wisdom is that one *can* learn sequentially and that doing so saves over fixed designs, but the field has heuristics, not an optimality theory: no one has pinned down the smallest possible regret, nor exhibited a rule provably attaining it.

## Baselines

**Thompson (1933) — posterior probability matching.** Faced with two treatments whose success probabilities $\tilde p_1, \tilde p_2$ are unknown, Thompson puts a uniform prior on each, and after observing $r_i$ successes and $s_i$ failures computes the posterior probability $P$ that treatment 1 is better than treatment 2 (an integral of Beta densities, $P^{(i)}_{p,p+dp} \propto \binom{n_i}{r_i} p^{r_i} q^{s_i}\,dp$). He proposes allocating a *fraction* $f_{(P)}$ of future individuals to treatment 1 — in proportion to the evidence that it is better — rather than committing immediately. His justification is the "saving of sacrifice": committing to one treatment costs expected sacrifice $1-P$ per future subject, whereas matching costs $P(1-P)+(1-P)P = 2PQ \le \tfrac12$, strictly less whenever a real preference exists. **Gap:** this is a Bayesian heuristic. It gives no frequentist guarantee, no rate at which the cost of learning grows with the horizon, and no statement that it is optimal — and it requires committing to a prior.

**Robbins (1952) — frequentist play-the-winner rules.** Robbins poses the maximize-$\mathbb{E}[S_n]$ problem distribution-free ($F,G$ known only to lie in a class) and studies concrete sampling rules, e.g. rule $R_1$: choose $A$ or $B$ at random for the first toss; thereafter stick with the same arm after a success and switch after a failure. He computes the resulting Markov chain's stationary head-probability $\lim_i p_i = \frac{\alpha+\beta-2\alpha\beta}{2-(\alpha+\beta)}$ and discusses when such "reasonably good solutions of the proper problems" beat fixed allocation. **Gap:** these are particular rules analyzed for their operating characteristics. There is no notion of the *best achievable* regret, no matching lower bound, and the rules are not shown to approach any optimum — Robbins explicitly says "optimum solutions to these problems are not known."

**Greedy and $\varepsilon$-greedy.** The obvious rule is to pull the arm with the largest current empirical mean. This can lock permanently onto an arm whose early samples happened to look good, never collecting enough data on the true best arm to correct — linear regret. The fix is to pull a uniformly random arm with probability $\varepsilon$ and the empirical best otherwise; but a *constant* $\varepsilon$ wastes a constant fraction of pulls on bad arms, again linear in $n$. **Gap:** to get sublinear regret the exploration probability must be annealed at exactly the right rate, and choosing that rate requires knowing the gaps in advance.

## Evaluation settings

The natural instance to reason on is a $K$-armed bandit, $K>1$, with reward distributions of bounded support (canonically $[0,1]$) or one-parameter parametric families (Bernoulli; Gaussian with known or unknown variance). The performance measure is the regret $R_n = \sum_i \Delta_i\mathbb{E}[T_i(n)]$ as a function of the horizon $n$, examined both asymptotically ($n\to\infty$, the growth rate and its leading constant) and, more demandingly, uniformly over all finite $n$. A rule is judged on two axes simultaneously: the *order* of regret in $n$ (constant? $\sqrt n$? $\log n$? linear?) and, once the order is fixed, the *leading constant* and how it depends on the gaps $\Delta_i$ and the divergences between arms. A second axis is robustness: whether the rule needs advance knowledge of the gaps, the horizon, or the noise scale.

## Code framework

The harness, the simulator, the empirical-mean bookkeeping, and the regret accounting are already in place; the arm-selection rule is the open slot.

```python
import numpy as np

class BanditEnv:
    """K arms; pulling arm i returns an i.i.d. reward with unknown mean mu[i]."""
    def __init__(self, means, rng):
        self.means = np.asarray(means, float)
        self.rng = rng
        self.mu_star = self.means.max()

    def pull(self, i):
        # bounded-support reward in [0,1]; here Bernoulli with mean mu[i]
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
    """Decide which arm to pull at round t from the current statistics.

    The rule may use only past pull counts, empirical means, and the
    round number; it has no access to the true means or gaps.
    """
    # TODO: choose the next arm to pull
    pass

def run(env, n, K):
    stats = ArmStats(K)
    regret = 0.0
    for t in range(1, n + 1):
        i = select_arm(stats, t)
        r = env.pull(i)
        stats.update(i, r)
        regret += env.mu_star - env.means[i]   # instance regret bookkeeping
    return regret
```
