We face the stochastic $K$-armed bandit: at each round $t$ we pull an arm $A_t \in \{1,\dots,K\}$ and collect a reward $X_t$ that is i.i.d. with unknown mean $\mu_a$ and bounded in $[0,1]$ (any bounded reward rescales). Writing $\Delta_a = \mu_* - \mu_a$ for the gap of arm $a$ and $N_a(n)$ for its pull count, the regret decomposes as $R_n = \sum_{a:\mu_a<\mu_*} \Delta_a\, \mathbb{E}[N_a(n)]$, a gap-weighted tally of sub-optimal pulls. Everything reduces to keeping $\mathbb{E}[N_a(n)]$ small for every bad arm, and I want to do it with an *index* policy — compute one number per arm from that arm's own data and pull the largest — because it is $O(K)$ per round, horizon-free, and trivial to deploy. The optimistic reading is the one I trust: don't act on the bare estimate $\hat\mu_a = S_a/N_a$, act on an upper confidence bound, so under-sampled arms get explored (wide bounds, high optimistic value) and well-sampled good arms get exploited (tight high bounds). The workhorse is UCB1, which pulls $\arg\max_a\, \hat\mu_a + \sqrt{2\log t / N_a}$ and enjoys $\mathbb{E}[R_n] \le \sum_a 8\log n/\Delta_a + (1+\pi^2/3)\sum_a \Delta_a$ — order-optimal in $n$, horizon-free, tuning-free, distribution-free over $[0,1]$. Four boxes ticked, but the constant is wrong, and the reason is structural. The Hoeffding bonus $\sqrt{2\log t/N_a}$ depends only on $N_a$ and the range $[0,1]$; it never looks at *where* $\hat\mu_a$ sits, so an arm at $\hat\mu=0.5$ and an arm at $\hat\mu=0.02$ get the same half-width at equal $N_a$. That is wrong: a coin landing heads two times in a hundred concentrates far harder than a fair coin, so its interval should be far narrower. UCB1 uses a symmetric, variance-blind width — the inverse of the quadratic Pinsker bound $2(p-q)^2 \le d(p,q)$, and Pinsker is loosest exactly near the boundaries $0$ and $1$. The information-theoretic floor of Lai and Robbins says any uniformly good policy obeys $\liminf_n \mathbb{E}[N_a(n)]/\log n \ge 1/d(\mu_a,\mu_*)$ with $d(p,q) = p\log(p/q) + (1-p)\log\frac{1-p}{1-q}$ the Bernoulli KL divergence; UCB1 pays $8/\Delta_a^2$ against a floor of $1/d(\mu_a,\mu_*)$, and since a Taylor expansion gives $d(p,q) = (p-q)^2/(2p(1-p)) + o((p-q)^2)$ the local curvature $1/(2p(1-p))$ blows up near the boundary while Pinsker keeps a flat $2$. So on low-reward arms — the rare-event regime of advertising or clinical trials — UCB1 over-explores badly. The whole loss is that its confidence width is the inverse of a quadratic when it should be the inverse of $d$ itself.

I propose KL-UCB: replace the additive Hoeffding bonus with a *KL ball*. The per-arm index at time $t$ is the largest mean whose Bernoulli-KL distance from the empirical mean fits inside a deviation budget,
$$U_a(t) = \max\{\, q \in [0,1] : N_a\, d(\hat\mu_a, q) \le c\log t \,\}, \qquad \hat\mu_a = S_a/N_a,$$
and we pull $\arg\max_a U_a(t)$. The budget $c\log t$ is the practical pure-log exploration function with default $c=1$; the theorem instead uses $\log t + 3\log\log t$, of which more below. The reason $d$ is the right currency is that it is the *exact* Chernoff large-deviation rate of a sample mean: running Chernoff's method on i.i.d. Bernoulli draws, $P(\hat\mu \ge \mu + \epsilon) = P(e^{\lambda \sum (X_t-\mu)} \ge e^{\lambda n \epsilon}) \le (\mu e^{\lambda(1-\mu-\epsilon)} + (1-\mu)e^{-\lambda(\mu+\epsilon)})^n$, and minimizing over $\lambda$ — optimizer $\lambda = \log\frac{(\mu+\epsilon)(1-\mu)}{\mu(1-\mu-\epsilon)}$ — collapses the base to exactly $e^{-d(\mu+\epsilon,\mu)}$. So $P(\hat\mu \ge \mu+\epsilon) \le e^{-n\,d(\mu+\epsilon,\mu)}$, the true rate, where Hoeffding would only give $e^{-2n\epsilon^2}$. Inverting that tail builds the bound: with a budget $\delta$, set $U = \max\{u : d(\hat\mu, u) \le \delta\}$; because $d(\hat\mu,\cdot)$ is zero at $\hat\mu$, strictly convex, and strictly increasing on $[\hat\mu,1]$, this is either the root $d(\hat\mu,U)=\delta$ or the endpoint $1$, and one shows $P(\mu > U) \le e^{-n\delta}$. As $\hat\mu$ rises toward $1$ the curve $d(\hat\mu,\cdot)$ steepens, so the width $U-\hat\mu$ self-tightens *for free* — exactly the variance adaptation UCB1 lacked, with no variance estimate to plug in. This is not a new family: relax $d$ to its Pinsker lower bound $2(p-q)^2$ and $N_a\,2(\hat\mu_a-q)^2 \le \log t$ solves to $q = \hat\mu_a + \sqrt{\log t/(2N_a)}$, recovering UCB1's additive bonus — UCB1 is precisely the Pinsker-relaxed special case of the same optimism.

What makes the method distribution-free even though $d$ is Bernoulli is a single convexity lemma. For any $X \in [0,1]$ with mean $\mu$, the function $f(x) = e^{\lambda x} - x(e^\lambda - 1) - 1$ is convex (second derivative $\lambda^2 e^{\lambda x} \ge 0$) and vanishes at $x=0$ and $x=1$; a convex function zero at both endpoints is $\le 0$ between them, so $e^{\lambda X} \le X(e^\lambda-1)+1$ pointwise, and taking expectations $\mathbb{E}[e^{\lambda X}] \le 1 - \mu + \mu e^\lambda$, which is exactly the Bernoulli MGF. The Bernoulli is therefore the *least concentrated* bounded law of a given mean — the worst case — and its rate $d$ dominates the deviations of every $[0,1]$ reward, so the Bernoulli-KL index is a valid upper confidence bound for all of them. This is the analogue of "variance $1/4$ is maximal on $[0,1]$," lifted from second moments to the whole MGF. The subtlety the regret proof must then handle is that $N_a(t)$ is random and adaptively chosen, so a naive union bound over $t$ would cost a factor $n$. The fix is a self-normalized supermartingale $W_t^\lambda = \exp(\lambda S(t) - N(t)\,\psi_\mu(\lambda))$ with $\psi_\mu(\lambda) = \log(1-\mu+\mu e^\lambda)$ and previsible pull-indicators $\epsilon_s$: the bounded-to-Bernoulli lemma makes $\mathbb{E}[W_{t+1}^\lambda \mid \mathcal{F}_t] \le W_t^\lambda$, so $\mathbb{E}[W_t^\lambda] \le 1$ despite the random $N(t)$. A geometric *peeling* of the range of $N(n)$ into blocks $t_k = \lfloor (1+\eta)^k\rfloor$ with $\eta = 1/(\delta-1)$, applying Markov on each slice with a $\lambda$ tuned to that block via the Legendre duality $d(z,\mu) = \sup_\lambda\{\lambda z - \psi_\mu(\lambda)\}$, yields the uniform deviation bound
$$P(u(n) < \mu) \le e\,\lceil \delta \log n\rceil\, e^{-\delta},$$
paying only a polynomial $\delta\log n$ factor in front of the exponential. Choosing $\delta = \log t + 3\log\log t$ makes the optimal-arm-underestimation term sum to $O(\log\log n)$ — the constant $c=3$ is what tames the series $\sum_t \lceil \log t^2 + c\log t\log\log t\rceil/(t(\log t)^c)$, and a smaller $c$ would let it diverge too fast for the proof to absorb. The regret then splits a sub-optimal arm's pulls into "the best arm was under-estimated" (controlled by the deviation bound) and "it wasn't but I pulled $a$ anyway"; the second piece re-indexes by sample count and splits at $K_n = \lfloor (1+\epsilon)(\log n + 3\log\log n)/d(\mu_a,\mu_*)\rfloor$, bounding the tail $s>K_n$ by a geometric Chernoff sum, giving $\mathbb{E}[N_n(a)] \le (1+\epsilon)\log n/d(\mu_a,\mu_*) + C_1\log\log n + C_2(\epsilon)/n^{\beta(\epsilon)}$ and hence $\limsup_n \mathbb{E}[R_n]/\log n \le \sum_{a:\mu_a<\mu_*} \Delta_a/d(\mu_a,\mu_*)$. For Bernoulli rewards that *is* the Lai-Robbins floor, so KL-UCB is asymptotically optimal in the binary case; and since the proof only ever used the Bernoulli MGF as an upper bound, the identical bound holds for every $[0,1]$ reward, improving UCB's constant by the full Pinsker gap, largest near the boundary.

The $3\log\log t$ correction is a theorem artifact: an $(1+\epsilon)\log t$ budget would also suffice, but for $\epsilon=0.1$ it only dominates $\log t + 3\log\log t$ past $t > 2\times10^{51}$, so in code I drop the $\log\log t$ term and use a pure $c\log t$ budget with $c=1$. The inner $\max$ has no closed form, but the convexity and monotonicity of $q \mapsto d(\hat\mu_a,q)$ on $[\hat\mu_a,1]$ make the bound the unique root of $d(\hat\mu_a,q) = (c\log t)/N_a$ (or the right endpoint), found by bisection seeded with the Pinsker upper bracket $\min(1, \hat\mu_a + \sqrt{(c\log t)/(2N_a)})$ — a handful of $d$-evaluations to tolerance, guarding the logarithms by clipping $\hat\mu_a$ and $q$ off the exact $0/1$ endpoints where $d = +\infty$. One generalization comes nearly free: because the proof only invoked an MGF bound, swapping $d$ for any one-parameter exponential family's Legendre rate $d(x,\mu(\theta)) = \sup_\lambda\{\lambda x - \log \mathbb{E}_\theta[e^{\lambda X}]\}$ — which equals that family's own KL divergence $\mu(\beta)(\beta-\theta) - b(\beta) + b(\theta)$ — gives Lai-Robbins optimality for that family, with $d(x,y) = x/y - 1 - \log(x/y)$ for exponential rewards, $d(x,y) = y - x + x\log(x/y)$ for Poisson, and the Gaussian fixed-variance case recovering UCB's additive $\sqrt{}$ bonus. An upper bound on the true $d$ still yields a valid, slightly looser policy, so the Bernoulli case is just the worst-case-bounded instance of a single principle.

```python
import math
import numpy as np

_EPS = 1e-15  # truncate means to [_EPS, 1 - _EPS]; d = +inf at the exact 0/1 endpoints


def kl_bernoulli(p, q):
    """Bernoulli KL divergence d(p, q) = p log(p/q) + (1-p) log((1-p)/(1-q))."""
    p = min(max(p, _EPS), 1 - _EPS)
    q = min(max(q, _EPS), 1 - _EPS)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def klucb(mu_hat, budget, kl=kl_bernoulli, upper=1.0, precision=1e-6, max_iter=50):
    """Generic KL-UCB inversion: max{ q in [mu_hat, upper] : kl(mu_hat, q) <= budget }.
    Bisection on the increasing branch q -> kl(mu_hat, q)."""
    lower = mu_hat
    for _ in range(max_iter):
        if upper - lower <= precision:
            break
        mid = 0.5 * (lower + upper)
        if kl(mu_hat, mid) > budget:
            upper = mid
        else:
            lower = mid
    return 0.5 * (lower + upper)


def klucb_bern(mu_hat, budget, precision=1e-6):
    """KL-UCB index for Bernoulli (and any [0,1]-bounded) rewards.
    Pinsker/Gaussian seed for the upper bracket: d(p,q) >= 2(p-q)^2 => U <= p + sqrt(budget/2)."""
    upper = min(1.0, mu_hat + math.sqrt(budget / 2.0))
    return klucb(mu_hat, budget, kl=kl_bernoulli, upper=upper, precision=precision)


class KLUCB:
    """KL-UCB index policy. Index for arm a at time t:
        U_a(t) = max{ q : N_a * d(mu_hat_a, q) <= c log t },   mu_hat_a = S_a / N_a.
    Pull argmax_a U_a(t); pull each unpulled arm first (index +inf)."""

    def __init__(self, K, c=1.0, tolerance=1e-4):
        self.K = K
        self.c = c                       # pure-log exploration multiplier; default c=1
        self.tolerance = tolerance
        self.counts = np.zeros(K)        # N_a
        self.rewards = np.zeros(K)       # S_a

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0

    def select_arm(self, t, context=None):
        if t < self.K:
            return t                      # round-robin init
        budget_t = self.c * math.log(max(t, 1))     # delta = c log t
        best_arm, best_index = 0, -math.inf
        for a in range(self.K):
            if self.counts[a] < 1:
                return a                  # never pulled -> index +inf
            mu_hat = self.rewards[a] / self.counts[a]
            index = klucb_bern(mu_hat, budget_t / self.counts[a], self.tolerance)  # U_a(t)
            if index > best_index:
                best_index, best_arm = index, a
        return best_arm

    def update(self, arm, reward, context=None):
        self.counts[arm] += 1
        self.rewards[arm] += reward


# Exponential-family rates: drop the matching d into `klucb(..., kl=...)`.
def kl_exponential(x, y):
    """d(x, y) = x/y - 1 - log(x/y) for exponential rewards."""
    if x <= 0 or y <= 0:
        return math.inf
    return x / y - 1.0 - math.log(x / y)


def kl_poisson(x, y):
    """d(x, y) = y - x + x log(x/y) for Poisson rewards."""
    y = max(y, _EPS)
    if x <= 0:
        return y
    return y - x + x * math.log(x / y)
```
