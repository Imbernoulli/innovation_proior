**Problem (from step 2).** Two index rungs left the contextual setting stuck at ~0.18 — UCB1 and
KL-UCB cannot use the context `x`, so they estimate marginal arm means when the best arm changes every
round. KL-UCB also over-explored the stochastic MAB (0.0612) with its front-loaded `c = 1` budget,
though it won non-stationary (0.0349). A deterministic confidence bonus is the wrong tool for both gaps.

**Key idea.** Replace deterministic optimism with Bayesian probability-matching: maintain a posterior
per arm, draw one plausible parameter from each, play the arm whose draw is largest. Exploration comes
from the posterior's own uncertainty — targeted, self-extinguishing, no horizon-tuned constant. Realize
it per regime: **Beta(1,1)** posterior for Bernoulli MABs; **LinTS** (a Gaussian posterior over each
arm's parameter vector `theta_a`) for the contextual world, sampling `theta_tilde_a` and scoring
`x·theta_tilde_a`; a **discounted** Beta posterior for non-stationarity.

**Why it works.** Sampling-and-argmax makes `P(play arm k) = P(arm k is optimal | data)`, giving
`O(log T / Delta_a)` regret (Agrawal–Goyal) and, at finite horizons, less front-loaded exploration than
UCB indices. LinTS's posterior width `v^2 B_a^{-1}` is large in unexercised context directions, so it
explores in context space — the structure the index rungs could not see. The discount weights recent
observations exponentially, giving an effective memory `~1/(1-gamma)` that forgets stale segments.

**Scaffold edit / hyperparameters.** One `BanditPolicy` with three branches selected by the harness.
Bernoulli branch: `alpha, beta` per arm from `Beta(1,1)`, sample `rng.beta(alpha, beta)`, argmax. LinTS
branch (`context_dim > 0`): ridge `lambda = 1`, sampling scale `v^2 = 0.25`, `B_inv` maintained by
Sherman–Morrison rank-one updates, `theta_tilde = theta_hat + L z` via Cholesky of `v^2 B_inv` (isotropic
fallback on failure). Non-stationary robustness: discount `gamma = 0.999` decays `alpha, beta` toward the
prior before each Bernoulli update, clamped `>= 1` so the posterior never collapses. Private RNG seeded
in `__init__`.

**What to watch.** Stochastic should recover to ~0.035 (below both prior rungs); contextual should
collapse from ~0.18 to ~0.02 (first rule to model `x·theta_a`); non-stationary should be competitive but
possibly a touch above KL-UCB's 0.0349, since 1000-round memory carries some stale mass into each new
segment. Strongest single rule across the three regimes — best on two of three, the only one that closes
the contextual gap.

```python
class BanditPolicy:
    """Thompson Sampling with Beta posterior for Bernoulli arms.

    For MAB: samples from Beta(alpha, beta) posterior per arm.
    For contextual bandits: uses Bayesian linear regression (LinTS)
    with Sherman-Morrison incremental inverse updates.
    For non-stationary: uses discounted posterior (gamma < 1).
    """

    def __init__(self, K: int, context_dim: int = 0):
        self.K = K
        self.context_dim = context_dim
        self.rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))

        # Beta posterior params for MAB (alpha=successes+1, beta=failures+1)
        self.alpha = np.ones(K, dtype=np.float64)
        self.beta_param = np.ones(K, dtype=np.float64)

        # Discount factor for non-stationary settings
        self._gamma = 0.999

        # LinTS parameters for contextual bandits
        if context_dim > 0:
            self._lambda = 1.0  # regularization
            self._v2 = 0.25  # sampling variance scale
            # B_inv_a via Sherman-Morrison updates
            self._B_inv = np.array([np.eye(context_dim) / self._lambda
                                    for _ in range(K)])
            self._f = np.zeros((K, context_dim), dtype=np.float64)
            self._theta_hat = np.zeros((K, context_dim), dtype=np.float64)

        # Tracking
        self.counts = np.zeros(K, dtype=np.float64)
        self.rewards = np.zeros(K, dtype=np.float64)

    def reset(self):
        self.alpha[:] = 1.0
        self.beta_param[:] = 1.0
        self.counts[:] = 0
        self.rewards[:] = 0
        if self.context_dim > 0:
            d = self.context_dim
            for a in range(self.K):
                self._B_inv[a] = np.eye(d) / self._lambda
                self._f[a] = np.zeros(d)
                self._theta_hat[a] = np.zeros(d)

    def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
        if context is not None and self.context_dim > 0:
            return self._lints_select(context)

        # Sample from Beta posterior for each arm
        samples = self.rng.beta(self.alpha, self.beta_param)
        return int(np.argmax(samples))

    def _lints_select(self, context: np.ndarray) -> int:
        """Linear Thompson Sampling for contextual bandits."""
        best_arm = 0
        best_val = -np.inf
        for a in range(self.K):
            mu_a = self._theta_hat[a]
            # Sample: theta ~ N(mu_a, v2 * B_inv_a)
            # Use Cholesky of B_inv for efficient sampling
            z = self.rng.standard_normal(self.context_dim)
            try:
                L = np.linalg.cholesky(self._v2 * self._B_inv[a])
                theta_sample = mu_a + L @ z
            except np.linalg.LinAlgError:
                theta_sample = mu_a + math.sqrt(self._v2) * z
            val = context @ theta_sample
            if val > best_val:
                best_val = val
                best_arm = a
        return best_arm

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        self.counts[arm] += 1
        self.rewards[arm] += reward

        if context is not None and self.context_dim > 0:
            # Sherman-Morrison update: B_inv -= (B_inv x x^T B_inv)/(1 + x^T B_inv x)
            Bx = self._B_inv[arm] @ context
            denom = 1.0 + context @ Bx
            self._B_inv[arm] -= np.outer(Bx, Bx) / denom
            self._f[arm] += reward * context
            self._theta_hat[arm] = self._B_inv[arm] @ self._f[arm]
        else:
            # Discounted Beta posterior update (for non-stationary robustness)
            self.alpha *= self._gamma
            self.beta_param *= self._gamma
            # Clamp to prevent posterior from collapsing
            self.alpha = np.maximum(self.alpha, 1.0)
            self.beta_param = np.maximum(self.beta_param, 1.0)
            # Update the pulled arm
            self.alpha[arm] += reward
            self.beta_param[arm] += (1.0 - reward)
```
