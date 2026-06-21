UCB1's measured numbers told me exactly where the optimism principle is right and where its *geometry* is wrong. On the stochastic MAB it landed at normalized regret $0.0378$ — solid, tight across seeds, the home regime working as the theory promised. On the contextual setting it was $0.179$, bad but for a reason already named: it ignores the context, so it estimates marginal arm means that are close, with no fixed best arm to converge to. And on the non-stationary setting it was its worst world at $0.083$, more than double its stochastic number — the slow-unlearning failure I predicted, where the full-history index is anchored by thousands of stale samples after each changepoint. The contextual loss is a *modeling* gap I cannot close with an index; the stochastic and non-stationary numbers are both about the *confidence geometry* of the index, and that is the defect to attack here. Look hard at the bonus $\sqrt{2\log t/N_a}$: it depends only on $N_a$ and the range $[0,1]$, never on *where* $\hat\mu_a$ sits, so an arm with $\hat\mu=0.5$ and one with $\hat\mu=0.05$ get the same half-width at equal counts. That is wrong — a coin landing heads five times in a hundred concentrates far harder than a fair coin, so its interval should be much narrower. UCB1's symmetric, range-only width is the inverse of the quadratic Pinsker bound, which is loosest precisely near $0$ and $1$ where these settings' arm means cluster, so it over-explores exactly the low- and high-mean arms and leaks regret on both Bernoulli worlds.

The floor I am trying to reach is the Lai–Robbins bound, generalized by Burnetas–Katehakis: for any uniformly-good policy $\liminf \mathbb{E}[N_a(T)]/\log T \ge 1/K_{\inf}(\nu_a,\mu^*)$, which for Bernoulli arms collapses to $1/d(\mu_a,\mu^*)$ with the Bernoulli KL $d(p,q)=p\log(p/q)+(1-p)\log((1-p)/(1-q))$. Compare this to what UCB1 pays. Pinsker says $d(\mu_a,\mu^*)\ge 2\Delta_a^2$, so $1/d \le 1/(2\Delta_a^2)$; UCB1's $8/\Delta_a^2$ is sixteen times that Pinsker-level constant, and the gap between $d$ and the Pinsker parabola *grows* near the boundaries — the local curvature of $d$ is $1/(2p(1-p))$, which blows up as $p\to 0$ or $1$ while Pinsker keeps the flat $2$. That is the quantitative version of "the boundary arm should be cheap to rule out, and UCB1 doesn't notice."

So I propose KL-UCB: keep the optimism, but build the confidence width from the *true* Bernoulli large-deviation rate $d$ rather than its Pinsker lower bound. The tail comes from Chernoff's method, not Hoeffding's: for i.i.d. Bernoulli with mean $\mu$, minimizing the moment generating function over the exponential tilt $\lambda$ — with the minimizer $\lambda=\log\frac{(\mu+\epsilon)(1-\mu)}{\mu(1-\mu-\epsilon)}$ collapsing the base — gives the *exact* rate $P(\hat\mu \ge \mu+\epsilon) \le \exp(-n\,d(\mu+\epsilon,\mu))$, whereas Hoeffding only delivers $\exp(-2n\epsilon^2)$, the same near $\mu=1/2$ but enormously weaker near the boundaries. So $n\,d(\hat\mu,\mu)$ is the natural "nats of surprise" in an observed deviation, and I threshold *that*. Inverting the tail, I define the index as the largest mean whose KL distance from $\hat\mu_a$ fits a deviation budget:

$$U_a(t) = \max\{\, q\in[0,1] : N_a\, d(\hat\mu_a, q) \le \log t \,\}.$$

Because $d(\hat\mu_a,\cdot)$ is zero at $q=\hat\mu_a$, strictly convex, and strictly increasing on $[\hat\mu_a,1]$, this is well defined — the unique root of $d(\hat\mu_a,q)=\log t/N_a$, or $1$ if the whole right branch is feasible. The payoff is automatic: as $\hat\mu_a$ rises toward $1$ the function $d(\hat\mu_a,\cdot)$ gets steeper, so the width $U_a(t)-\hat\mu_a$ *shrinks on its own*. The interval is asymmetric and self-tightening near the boundary, for free, with no variance estimate to plug in. And UCB1 is exactly the special case where I replace $d(p,q)$ by its Pinsker lower bound $2(p-q)^2$: then $N_a\cdot 2(\hat\mu_a-q)^2 \le \log t$ solves to $q=\hat\mu_a+\sqrt{\log t/(2N_a)}$, the additive sqrt bonus. So I am not inventing a new family — I am using the *tight* member of the one UCB1 already lives in.

The budget is set to make the failure risk decay like $1/t$: I want $\exp(-N_a\cdot\text{budget}/N_a)\sim 1/t$, i.e. budget $\sim\log t$. The proven exploration function is $f(t)=\log t + 3\log\log t$, where the $3\log\log t$ makes the optimal-arm under-estimation term provably negligible, but that term is a theorem artifact: for it to matter against a clean $(1+\epsilon)\log t$ you would need $t$ past $\sim 10^{51}$, far beyond $T=10000$. So the practical budget is the pure $c_{\text{impl}}\log t$ with $c_{\text{impl}}=1$, the theorem-tight constant, and I drop the $\log\log$ correction. The full proof — a self-normalized supermartingale $W_t=\exp(\lambda S(t)-N(t)\psi_\mu(\lambda))$ to handle the *random* number of pulls, peeled over geometric slices of $N(t)$, giving $P(\text{under-estimate}) \le e\lceil \text{budget}\log T\rceil \exp(-\text{budget})$, then decomposing suboptimal pulls into best-arm under-estimates and an over-estimate tail — delivers $\mathbb{E}[N_a(T)] \le (1+\epsilon)\log T/d(\mu_a,\mu^*) + O(\log\log T)$, hence $\limsup R_T/\log T \le \sum_a \Delta_a/d(\mu_a,\mu^*)$, the Lai–Robbins floor. For both Bernoulli worlds this is asymptotically optimal, a strict leading-constant improvement over UCB1, and it stays distribution-free over $[0,1]$: the convexity lemma $\mathbb{E}[\exp(\lambda X)] \le 1-\mu+\mu\exp(\lambda)$ shows the Bernoulli is the least-concentrated bounded law, so its $d$ dominates every bounded distribution's deviations and the proof never needed binary rewards.

Two harness decisions follow from UCB1's failures. First, KL-UCB does *not* by itself forget, and the natural non-stationary move — a sliding window over the last $W$ pulls — is the same trap as before: windowing keeps every confidence radius permanently wide, which inflates the *stationary* regret, exactly the world I am here to *tighten*, not loosen. Since one rule is graded on all three and I cannot see which I am in, I keep KL-UCB on the full history with no window; any non-stationary improvement must come purely from the tighter index relearning a changed arm faster than UCB1's range-only bound, not from forgetting. Second, KL-UCB has no contextual machinery either, so on the contextual world it again runs per-arm KL-UCB ignoring $x$, and I expect no improvement there — the modeling gap is untouched. For the implementation I must compute $U_a(t)$, which has no closed form since $d$ cannot be inverted in its second argument, but the geometry is friendly: $q\mapsto d(p,q)$ is strictly convex, zero at $p$, increasing on $[p,1]$, so the equation has a unique root I find by bisection on $[p,1)$. The harness exposes `kl_ucb_bound` via `scipy.optimize.brentq`, but I hand-write a fixed 32-iteration binary search instead — 32 halvings give $\sim 10^{-10}$ precision at deterministic cost and avoid the per-call `brentq` overhead, which matters because this runs $K$ times per round over $T=10000$ rounds and many seeds. Each iteration computes $d(p,\text{mid})$ directly, with $p$ clipped off the exact $0/1$ endpoints (where $d=+\infty$) by a tiny epsilon so the logarithm never sees zero, and I seed the bracket on $[p,1-10^{-10}]$ rather than a Pinsker bracket since the fixed iteration budget converges fully either way. The policy is then: round-robin each arm once, then for each arm compute the KL-UCB index from its running $\hat\mu_a$ and $N_a$ and argmax; `update` just accumulates counts and reward sums — no buffer, no window, no context state. I expect the non-stationary number to drop below $0.083$, the contextual to sit near $0.18$, and — honestly — the stochastic number may come in *worse* than $0.0378$, because the $c=1$ budget explores more conservatively per arm early and at $T=10000$ the asymptotic constant advantage may not have materialized.

```python
class BanditPolicy:
    """KL-UCB: Kullback-Leibler Upper Confidence Bound.

    Vanilla KL-UCB per Garivier & Cappe 2011.  Index for arm a at time t is:
        U_a(t) = sup { q in [0,1] : N_a(t) * kl(mu_hat_a, q) <= c*log(t) }
    with c = 1 (theorem-tight constant) and kl the Bernoulli KL divergence.

    Implements the Bernoulli KL-UCB index formula used by SMPyBandits; no
    sliding window (which harms the stationary stochastic regime).
    """

    def __init__(self, K: int, context_dim: int = 0):
        self.K = K
        self.context_dim = context_dim
        self.counts = np.zeros(K, dtype=np.float64)
        self.rewards = np.zeros(K, dtype=np.float64)

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0

    @staticmethod
    def _fast_kl_ucb(p: float, n: int, t: int) -> float:
        """Fast KL-UCB bound via binary search (no scipy dependency)."""
        if n == 0:
            return 1.0
        p = max(min(p, 1 - 1e-10), 1e-10)
        threshold = math.log(max(t, 1)) / n
        lo, hi = p, 1.0 - 1e-10
        for _ in range(32):  # 32 iterations gives ~1e-10 precision
            mid = (lo + hi) * 0.5
            # KL(Bernoulli(p) || Bernoulli(mid))
            kl = p * math.log(p / mid) + (1 - p) * math.log((1 - p) / (1 - mid))
            if kl < threshold:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
        # Initial round-robin: each arm once.
        if t < self.K:
            return t

        # Standard KL-UCB index for each arm (no sliding window).
        best_arm = 0
        best_idx = -1e100
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
            mu_hat = self.rewards[a] / self.counts[a]
            idx = self._fast_kl_ucb(mu_hat, int(self.counts[a]), t + 1)
            if idx > best_idx:
                best_idx = idx
                best_arm = a
        return best_arm

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        self.counts[arm] += 1
        self.rewards[arm] += reward
```
