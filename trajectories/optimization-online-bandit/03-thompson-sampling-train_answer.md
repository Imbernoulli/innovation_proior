KL-UCB's numbers split exactly along the fault line I had predicted. On the non-stationary setting it dropped to $0.0349$ from UCB1's $0.083$ — a clean win, confirming that the tighter KL index relearns each changepoint faster on full history even without forgetting. On the contextual setting it was $0.179$, essentially identical to UCB1's $0.179$ — the modeling gap was untouched, context still ignored. But on the stochastic MAB it got *worse*, $0.0612$ versus UCB1's $0.0378$: the risk I had flagged materialized, because with the $c=1$ log-budget KL-UCB explores more conservatively per arm early, and at $T=10000$ the asymptotic constant advantage never arrives — the heavier early exploration costs more than the tighter constant saves. So two things are now forced. An index policy, Hoeffding or KL, *cannot* use the context, so the $0.18$ will sit there until I change what is modeled; and I want exploration that is not tied to a single deviation constant tuned against the horizon. Both point away from deterministic confidence bonuses and toward a Bayesian, randomized rule.

What I propose is Thompson Sampling, realized per regime. The idea is not "another index": instead of adding an exploration term that is a fixed function of the counts, I make exploration come from the *posterior's own uncertainty*. Put a prior on each arm's unknown parameter, update it to a posterior as rewards arrive, and at each round draw one plausible value from each arm's posterior and play the arm whose draw is largest. Exploration is then automatically targeted — an arm I am unsure about has a wide posterior and its draw can come out high, so I sample it; an arm I am confident is bad has a tight posterior near its low mean and its draw essentially never wins, so I leave it alone — and as posteriors concentrate the exploration extinguishes itself with no schedule to tune. The defining equivalence is that the probability I play arm $k$ equals the posterior probability that arm $k$ is optimal: probability-matching realized by sampling. Crucially I sample the *parameter* (the plausible mean), not a $0/1$ outcome, because the question is which arm *could be best*, which is a question about means.

For the Bernoulli worlds the posterior is conjugate and trivial. The uniform prior on $[0,1]$ is $\text{Beta}(1,1)$; after $r$ successes and $s$ failures the posterior is $\text{Beta}(r+1, s+1)$ — the data enter through the counts, the prior contributes the $+1$'s, and an observation just bumps one parameter. So I maintain $\alpha_a$ (successes $+1$) and $\beta_a$ (failures $+1$) per arm, draw $\theta_a\sim\text{Beta}(\alpha_a,\beta_a)$, and play $\arg\max_a \theta_a$. The Agrawal–Goyal analysis gives $\mathbb{E}[N_a(T)]=O(\log T/\Delta_a^2 + 1/\Delta_a^4)$ and hence $O(\log T/\Delta_a)$ regret, matching the Lai–Robbins order; and at finite horizons the randomization avoids the front-loaded deterministic exploration that made KL-UCB's $c=1$ budget over-explore the stochastic MAB. That front-loading is exactly the disease a posterior-sampling rule does not have, so on the stochastic world I expect Beta-Bernoulli TS to recover and beat both prior rungs.

The contextual setting is the gap that has refused to move, and here the reward is $x\cdot\theta_a$ for a $d=10$ parameter vector $\theta_a$ per arm, with $x$ changing every round, so there is no fixed best arm. A per-arm scalar posterior cannot represent this; I need a posterior over each arm's *parameter vector* and to score a plausible draw against the current context. This is linear Thompson sampling: model arm $a$'s rewards as $x\cdot\theta_a + \text{noise}$ with a Gaussian likelihood and Gaussian prior, so the posterior is Gaussian, $\theta_a\sim\mathcal{N}(\hat\theta_a, v^2 B_a^{-1})$, where $B_a=\lambda I + \sum xx^\top$ is the regularized design matrix over the contexts on which arm $a$ was played, $\hat\theta_a = B_a^{-1}f_a$ is the ridge estimate with $f_a=\sum \text{reward}\cdot x$, and $v^2$ scales the sampling covariance. Each round I draw $\tilde\theta_a\sim\mathcal{N}(\hat\theta_a, v^2 B_a^{-1})$ per arm and play $\arg\max_a x\cdot\tilde\theta_a$. The posterior width $v^2 B_a^{-1}$ is large in directions the arm has not been exercised, so exploration is targeted in *context space* — precisely the structure the index rungs could not see. Two choices keep this cheap. I never form $B_a$ and invert it; I maintain $B_a^{-1}$ directly with the Sherman–Morrison rank-one update $B_a^{-1}\leftarrow B_a^{-1} - (B_a^{-1}xx^\top B_a^{-1})/(1+x^\top B_a^{-1}x)$, an $O(d^2)$ update with no inversion, and recompute $\hat\theta_a = B_a^{-1}f_a$ after bumping $f_a$. And to sample I take a Cholesky factor $L$ of $v^2 B_a^{-1}$ and set $\tilde\theta_a=\hat\theta_a + Lz$ with $z$ standard normal — exact, $O(d^3)$ but cheap at $d=10$ — falling back to isotropic $\hat\theta_a + \sqrt{v^2}\,z$ if Cholesky fails. The regularizer is $\lambda=1$ (a well-conditioned unit ridge prior at $d=10$) and the sampling-variance scale is $v^2=0.25$: deliberately modest, because the contextual noise std is only $0.1$ and rewards are clipped to $[0,1]$, so a large $v^2$ would over-explore and a too-small one under-explore — $0.25$ is the conservative middle matching the $[0,1]$-bounded variance proxy.

The non-stationary setting needs a third mechanism, because the danger with any posterior method on a piecewise-stationary world is the same staleness that hurt UCB1: after a changepoint the posterior is dominated by thousands of old observations, sharply and wrongly concentrated, slow to overcome with fresh data. The fix that needs no changepoint detection is a *discounted* posterior — before each Bernoulli update, decay both parameters toward the prior, $\alpha\leftarrow\gamma\alpha$, $\beta\leftarrow\gamma\beta$, then add the new observation. Geometrically this weights recent observations exponentially more, giving an effective memory $\sim 1/(1-\gamma)$ rounds, so a changepoint's stale mass decays on that timescale. With $\gamma=0.999$ the effective window is $\sim 1000$ rounds — comfortably shorter than the $2000$-round segments between changepoints, so each segment is learned and the previous one forgotten before the next change, yet long enough that the posterior concentrates within a segment. I clamp $\alpha,\beta\ge 1$ after decaying so the posterior can never collapse below the uniform prior, which keeps a floor of exploration alive even on long stationary stretches. That clamp is what lets me run the discount *globally* on the Bernoulli branch without wrecking the stochastic world: there the means never change, so the discount only mildly inflates posterior width, which a posterior-sampling rule tolerates far better than KL-UCB tolerated its $c=1$ budget. The contextual branch gets no discount — LinTS's ridge accumulation has its own regularization and the contextual world is stationary, so discounting there would only add noise.

The whole rule is one `BanditPolicy` with three branches the harness selects automatically: `select_arm` dispatches to the LinTS branch when `context is not None and context_dim > 0`, and otherwise samples Beta. One detail I have to get right is the RNG — the arm draws must be genuinely random for probability-matching to work — so I seed a private `np.random.default_rng` in `__init__` rather than relying on the global `np.random` the placeholder used, and `reset` restores $\alpha,\beta$ to $1$, zeros counts, and re-initializes the LinTS state to the prior. I expect Beta-Bernoulli TS around $0.035$ on the stochastic world (below both prior rungs), a collapse from $\sim 0.18$ to roughly $0.02$ on contextual (the first rule to model $x\cdot\theta_a$), and a competitive non-stationary number plausibly a touch above KL-UCB's $0.0349$, since a $1000$-round memory still carries some stale mass into each new segment. That makes it the strongest single rule across the three regimes — best on two of three and the only one to close the contextual gap — because it adapts its mechanism per regime instead of forcing one index everywhere.

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
