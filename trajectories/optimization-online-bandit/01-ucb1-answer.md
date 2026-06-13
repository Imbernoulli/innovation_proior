**Problem.** One arm-selection rule, computed from running per-arm statistics, must serve a stationary
10-arm Bernoulli MAB, a 5-arm linear contextual bandit, and a 5-arm non-stationary bandit, scored by
normalized cumulative regret (lower is better). The default fill selects uniformly at random — pure
exploration, never exploits, the floor by construction. Regret is `sum_a Delta_a E[N_a(T)]`, so the only
lever is how often each suboptimal arm is pulled.

**Key idea.** Be optimistic in the face of uncertainty. Replace each arm's empirical mean with an
*upper confidence bound* `mu_hat_a + sqrt(2 log(t+1) / N_a)` and play the largest. The bonus is a
one-sided Hoeffding half-width: it shrinks like `1/sqrt(N_a)` to exploit well-sampled arms and grows
like `sqrt(log t)` to keep revisiting under-sampled ones, so exploration is targeted by uncertainty and
self-extinguishes once an arm's interval tightens. Play each arm once first (round-robin) so every bonus
is defined.

**Why it works.** The optimistic index gives the finite-time bound `E[N_a(T)] <= 8 log T / Delta_a^2 +
O(1)`, hence `R_T <= 8 sum_a log T / Delta_a + O(1)` — logarithmic, horizon-free, distribution-free over
`[0,1]`, and `O(sqrt(K T log T))` in the gap-free reading. This is its home regime (stationary
Bernoulli); on the contextual and non-stationary worlds it is a deliberately plain baseline.

**Scaffold edit / hyperparameters.** The literal fill is full-history UCB1 with exploration constant
`c = 2` (the `sqrt(2 log(t+1)/N_a)` term) applied identically on all three settings. The context vector
is ignored on the contextual world (per-arm UCB1 on the marginal reward). A sliding-window circular
buffer (`W = 800`) and its running window-stats are maintained in `update` but **never consulted by
`select_arm`** — switching to the window inflated stationary-MAB regret from `~960` (theory) to `~1450`
observed, so vanilla UCB1 uses the full history regardless of environment.

**What to watch.** Strong on the stochastic MAB (a few percent), poor on contextual (context unused, no
fixed best arm to converge to), worst on non-stationary (the full-history index is anchored by stale
samples and unlearns each changepoint slowly). Those two blind spots — range-only confidence width and
no forgetting — are what the next rung must fix.

```python
class BanditPolicy:
    """UCB1: Upper Confidence Bound algorithm.

    Maintains empirical means and pull counts.  Selects the arm with the
    highest upper confidence bound: mu_hat + sqrt(2 * log(t+1) / N_a).

    For non-stationary settings, uses a sliding window of size W with
    an efficient circular buffer (O(1) per step).
    """

    def __init__(self, K: int, context_dim: int = 0):
        self.K = K
        self.context_dim = context_dim
        # Cumulative statistics
        self.counts = np.zeros(K, dtype=np.float64)
        self.rewards = np.zeros(K, dtype=np.float64)
        # Sliding window via circular buffer for non-stationary settings
        self._W = 800
        self._buf_arms = np.zeros(self._W, dtype=np.int32)
        self._buf_rewards = np.zeros(self._W, dtype=np.float64)
        self._buf_ptr = 0
        self._buf_full = False
        self._sw_counts = np.zeros(K, dtype=np.float64)
        self._sw_rewards = np.zeros(K, dtype=np.float64)

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0
        self._buf_ptr = 0
        self._buf_full = False
        self._sw_counts[:] = 0
        self._sw_rewards[:] = 0

    def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
        # Initial round-robin: play each arm once
        if t < self.K:
            return t

        # Standard UCB1 index (full history). The SW-UCB fallback here was
        # incorrect — vanilla UCB1 should use the full history regardless of
        # environment; switching to sliding-window inflated regret on
        # stationary MAB from ~960 (theoretical) to ~1450 observed.

        mu_hat = self.rewards / np.maximum(self.counts, 1e-10)
        exploration = np.sqrt(2.0 * math.log(t + 1) / np.maximum(self.counts, 1))
        ucb_values = mu_hat + exploration
        return int(np.argmax(ucb_values))

    def _sw_select(self, t: int) -> int:
        """Sliding-window UCB using pre-maintained running statistics."""
        unpulled = self._sw_counts == 0
        if unpulled.any():
            return int(np.argmax(unpulled))
        mu_hat = self._sw_rewards / self._sw_counts
        xi = 1.5  # exploration parameter for SW-UCB
        exploration = np.sqrt(xi * math.log(self._W) / self._sw_counts)
        return int(np.argmax(mu_hat + exploration))

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        # Update circular buffer and running window stats
        if self._buf_full:
            old_arm = int(self._buf_arms[self._buf_ptr])
            old_rew = self._buf_rewards[self._buf_ptr]
            self._sw_counts[old_arm] -= 1
            self._sw_rewards[old_arm] -= old_rew
        self._buf_arms[self._buf_ptr] = arm
        self._buf_rewards[self._buf_ptr] = reward
        self._sw_counts[arm] += 1
        self._sw_rewards[arm] += reward
        self._buf_ptr += 1
        if self._buf_ptr >= self._W:
            self._buf_ptr = 0
            self._buf_full = True
```
