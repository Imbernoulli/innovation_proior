**Problem (from step 1).** UCB1 was strong on the stochastic MAB (0.0378) but its confidence bonus
`sqrt(2 log t / N_a)` is *range-only* — it ignores where `mu_hat_a` sits, so a near-boundary arm and a
fair-coin arm get the same width at equal counts. That symmetric Hoeffding radius is the inverse of the
Pinsker bound `2(p-q)^2`, loosest exactly near 0 and 1 where these settings' arm means cluster, costing
the `8/Delta_a^2` constant against the Lai–Robbins floor `1/d(mu_a, mu*)`. Non-stationary (0.083) was
its worst world.

**Key idea.** Keep optimism, sharpen the geometry: build the confidence width from the *true* Bernoulli
large-deviation rate `d(p,q) = p log(p/q) + (1-p) log((1-p)/(1-q))` instead of its quadratic Pinsker
lower bound. The index is the KL ball `U_a(t) = max{ q in [0,1] : N_a * d(mu_hat_a, q) <= log t }`, the
largest mean whose KL distance from `mu_hat_a` fits the deviation budget. It self-tightens as `mu_hat_a`
approaches 0 or 1 (where `d` is steep), for free; UCB1 is the Pinsker-relaxed special case.

**Why it works.** The same proof, run on `d` rather than `2(p-q)^2`, gives `E[N_a(T)] <=
(1+eps) log T / d(mu_a, mu*) + O(log log T)`, matching the Lai–Robbins floor for Bernoulli — asymptotic
optimality and a strict leading-constant gain over UCB1, distribution-free over `[0,1]` (the Bernoulli
is the least-concentrated bounded law, so its `d` dominates all deviations).

**Scaffold edit / hyperparameters.** Vanilla KL-UCB, exploration constant `c = 1` (budget `log(t+1)`),
**no** `3 log log t` correction (a theorem artifact irrelevant below `t ~ 10^51`). The KL ball is
inverted by a fixed 32-iteration binary search on `[mu_hat_a, 1)` (≈1e-10 precision, deterministic,
avoids `scipy.brentq` overhead), with `mu_hat_a` clipped off 0/1. Round-robin each arm once. **No
sliding window** — windowing inflates the stationary regret it just improved — so the index uses the
full history on all three settings; the context is ignored on the contextual world (same modeling gap as
UCB1).

**What to watch.** Non-stationary should drop below UCB1's 0.083 as the tighter index relearns each
changepoint faster on full history. Stochastic may not beat 0.0378: the `c = 1` budget explores more
conservatively early, and the asymptotic constant advantage may not materialize at `T = 10000`.
Contextual stays near 0.18 — context still unused, forcing a context-modeling rule next.

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
