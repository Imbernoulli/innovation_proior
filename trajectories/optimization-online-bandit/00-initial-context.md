## Research question

Online sequential decision-making under the exploration–exploitation tension: at each of `T` rounds
an agent picks one of `K` arms, observes a single stochastic reward for the arm it pulled (and nothing
about the arms it did not), and wants to do nearly as well as an oracle that always plays the best arm.
The honest yardstick is **regret** — the cumulative shortfall against that oracle. The single thing
being designed is the **arm-selection rule**: how to convert running per-arm statistics into the next
pull so that exploration is spent only where the value is genuinely uncertain and stops once it is
resolved. Everything else — the arms, the environments, the evaluation loop — is fixed.

What makes this task more than a single textbook bandit is that the *same* rule is graded on three
structurally different worlds at once: a stationary 10-arm Bernoulli MAB, a 5-arm linear **contextual**
bandit where the reward depends on a per-round feature vector, and a **non-stationary** 5-arm Bernoulli
bandit whose best arm jumps at four abrupt changepoints. A rule that is optimal on one of these can be
badly miscalibrated on the others, so the design pressure is to pick an exploration mechanism that
either generalizes across the three regimes or is cheap to specialize per regime.

## Prior art before the first rung (sequential-allocation lineage)

The first rung — UCB1 — is itself the resolution of a line of sequential-allocation ideas. These are
the methods that precede the ladder; the index rule the first rung lands is what they converged to.

- **Greedy / certainty-equivalence.** Estimate each arm's mean by its empirical average and always
  play the largest. Cheap, but a single unlucky early streak on the truly-best arm buries its estimate;
  because greedy never pulls it again, the estimate never recovers and the agent locks onto a worse arm
  forever — regret linear in `T`. Gap: a noisy estimate can permanently exile a good arm.
- **ε-greedy (Robbins 1952 lineage; Watkins 1989).** Play greedy but with probability ε pick a uniform
  random arm. This breaks lock-in, but a constant ε spends a constant fraction of all rounds on
  *uniformly* random arms forever — gap-sized regret on `~εT` rounds, still linear. Decaying ε to get
  sublinear regret requires tuning the decay rate, which in turn needs a lower bound on the smallest
  gap the agent does not know. Gap: undirected exploration wastes pulls on arms already known bad.
- **Lai–Robbins index policies (1985); Agrawal (1995).** Lai and Robbins proved the asymptotic regret
  floor — a suboptimal arm must be pulled at least `~(log T)/D(p_a‖p*)` times, governed by a
  Kullback–Leibler divergence — and constructed index rules attaining it, which Agrawal made cheap by
  writing the index as a simple function of the rewards seen so far. Gap: the guarantees were
  asymptotic; nothing pinned a confidence radius that holds at every finite horizon.

These set up exactly what the first rung must supply: exploration *directed by uncertainty*,
self-extinguishing once the uncertainty is resolved, with a finite-time guarantee, computed by an
`O(K)`-per-round index over running sufficient statistics.

## The fixed substrate

A single evaluation harness is frozen and must not be touched. It supplies, in `custom_bandit.py`:

- **Three environments** behind one `make_env` factory. `StochasticMAB`: `K=10` Bernoulli arms with
  fixed means `[0.10,0.20,0.30,0.35,0.40,0.50,0.55,0.60,0.70,0.80]`, `T=10000`, no context.
  `ContextualBandit`: `K=5`, `d=10`; each round draws a unit-sphere context `x`, expected reward of arm
  `a` is `x·θ_a` for fixed unit-scaled `θ_a`, Gaussian noise std `0.1`, rewards clipped to `[0,1]`,
  `T=10000`. `NonStationaryMAB`: `K=5` Bernoulli arms with four changepoints at `{2000,4000,6000,8000}`,
  the best arm cycling through arms 0→1→2→3→4, `T=10000`.
- **The evaluation loop** `run_bandit(env, policy, horizon)`: each round it draws the context
  (`None` for the two MABs), calls `policy.select_arm(t, context)`, pulls, accumulates instantaneous
  regret, then calls `policy.update(arm, reward, context)`. The reported metric is
  **normalized cumulative regret** `= cumulative_regret / T` (lower is better).
- **Two KL utilities** a policy may call: `kl_bernoulli(p, q)` (Bernoulli KL via `scipy.special.rel_entr`,
  edge-clipped) and `kl_ucb_bound(mu_hat, n, t, c)` (the KL-UCB upper bound via `scipy.optimize.brentq`).

The policy sees only `(t, context)` to choose and `(arm, reward, context)` to update; it never sees the
arm means, the changepoints, or which environment it is in. One rule must serve all three.

## The editable interface

Exactly one region is editable — the `BanditPolicy` class in `custom_bandit.py` (lines 261–321 of the
scaffold). Every method on the ladder is a fill of this same contract:

- `__init__(K, context_dim)` — initialize state for `K` arms, with `context_dim>0` only on the
  contextual setting.
- `reset()` — clear state for a new run.
- `select_arm(t, context) -> int` — choose an arm in `{0,…,K-1}`.
- `update(arm, reward, context)` — fold the observed reward into the state.

The starting point is the scaffold default: track per-arm pull counts and reward sums, but **select
uniformly at random** — no exploration mechanism at all. Each method on the ladder replaces exactly
this class and nothing else.

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

Each policy is run on all three environments — **stochastic_mab**, **contextual**, **nonstationary** —
each over three seeds `{42, 123, 456}`, for `T=10000` rounds. The reported metric per environment is
**normalized cumulative regret** (`cumulative_regret / T`); **lower is better** on all three. The three
columns are reported separately, so a rule is judged on its behaviour across the stationary, contextual,
and non-stationary regimes simultaneously.
