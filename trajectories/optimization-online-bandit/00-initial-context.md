## Research question

Online sequential decision-making under the exploration–exploitation tension: at each of `T` rounds an agent picks one of `K` arms, observes a single stochastic reward for the arm it pulled, and wants to do nearly as well as an oracle that always plays the best arm. The honest yardstick is **regret** — the cumulative shortfall against that oracle. The only design choice is the **arm-selection rule**.

The same rule is graded on three structurally different worlds: a stationary 10-arm Bernoulli MAB, a 5-arm linear **contextual** bandit where reward depends on a per-round feature vector, and a **non-stationary** 5-arm Bernoulli bandit whose best arm jumps at four abrupt changepoints.

## Prior art / Background / Baselines

- **Greedy / certainty-equivalence.** Estimate each arm's mean by its empirical average and always play the largest.
- **ε-greedy.** Mostly play the empirical-best arm, but with probability ε pick a uniformly random arm.
- **Lai–Robbins index policies.** Compute an index from each arm's observed rewards and play the arm with the largest index; these policies achieve asymptotically optimal regret.

## Fixed substrate / Code framework

The evaluation harness in `custom_bandit.py` is frozen and must not be touched:

- **Three environments** behind one `make_env` factory. `StochasticMAB`: `K=10` Bernoulli arms with fixed means `[0.10,0.20,0.30,0.35,0.40,0.50,0.55,0.60,0.70,0.80]`, `T=10000`, no context. `ContextualBandit`: `K=5`, `d=10`; each round draws a unit-sphere context `x`, expected reward of arm `a` is `x·θ_a` for fixed unit-scaled `θ_a`, Gaussian noise std `0.1`, rewards clipped to `[0,1]`, `T=10000`. `NonStationaryMAB`: `K=5` Bernoulli arms with changepoints at `{2000,4000,6000,8000}`, best arm cycling through arms 0→1→2→3→4, `T=10000`.
- **The evaluation loop** `run_bandit(env, policy, horizon)`: each round it draws the context (`None` for the two MABs), calls `policy.select_arm(t, context)`, pulls, accumulates instantaneous regret, then calls `policy.update(arm, reward, context)`. The reported metric is **normalized cumulative regret** `= cumulative_regret / T` (lower is better).
- **Two KL utilities** a policy may call: `kl_bernoulli(p, q)` (Bernoulli KL via `scipy.special.rel_entr`, edge-clipped) and `kl_ucb_bound(mu_hat, n, t, c)` (the KL-UCB upper bound via `scipy.optimize.brentq`).

The policy sees only `(t, context)` to choose and `(arm, reward, context)` to update; it never sees the arm means, the changepoints, or which environment it is in.

## Editable interface

Exactly one region is editable — the `BanditPolicy` class in `custom_bandit.py`. Every method fills the same contract:

- `__init__(K, context_dim)` — initialize state for `K` arms, with `context_dim>0` only on the contextual setting.
- `reset()` — clear state for a new run.
- `select_arm(t, context) -> int` — choose an arm in `{0,…,K-1}`.
- `update(arm, reward, context)` — fold the observed reward into the state.

The starting point is the scaffold default: track per-arm pull counts and reward sums, but **select uniformly at random** — no exploration mechanism at all.

```python
# EDITABLE region of custom_bandit.py — default fill (uniform-random arm selection)
class BanditPolicy:
    """Bandit policy: the agent's exploration-exploitation strategy.

    The evaluation loop calls:
        policy = BanditPolicy(K, context_dim)
        policy.reset()
        for t in range(T):
            context = env.get_context()          # None for MAB
            arm = policy.select_arm(t, context)  # choose arm
            reward, _ = env.pull(arm)
            policy.update(arm, reward, context)  # observe reward

    Available utilities (fixed, importable):
        kl_bernoulli(p, q)              : KL divergence between Bernoulli(p) and Bernoulli(q)
        kl_ucb_bound(mu_hat, n, t, c)   : KL-UCB upper confidence bound
    """

    def __init__(self, K: int, context_dim: int = 0):
        self.K = K
        self.context_dim = context_dim
        self.counts = np.zeros(K, dtype=np.float64)
        self.rewards = np.zeros(K, dtype=np.float64)

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0

    def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
        # Placeholder: uniform random — replace with your algorithm
        return int(np.random.randint(self.K))

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        self.counts[arm] += 1
        self.rewards[arm] += reward
```

## Evaluation settings

Each policy is run on all three environments — **stochastic_mab**, **contextual**, **nonstationary** — each over three seeds `{42, 123, 456}`, for `T=10000` rounds. The reported metric per environment is **normalized cumulative regret** (`cumulative_regret / T`); **lower is better** on all three. The three columns are reported separately, so a rule is judged on its behaviour across the stationary, contextual, and non-stationary regimes simultaneously.
