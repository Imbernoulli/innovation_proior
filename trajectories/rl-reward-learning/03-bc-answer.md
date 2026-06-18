**Problem (from step 2).** AIRL beat GAIL everywhere but its terminating-body seeds were
two-dead-one-alive (Hopper {153, 169, 2198}; Walker2d {290, 654, 129}): the adversarial reward is
*non-stationary by construction*, and on clean, tight expert demos the discriminator's pull leaves most
seeds stuck in degenerate basins before a usable reward emerges. Two rungs of normalization and shaping
tamed it but did not make the min-max game reliable. The question for the top rung: does a stable
supervised fit beat the principled-but-unstable adversarial methods on clean demos?

**Key idea.** Skip reward learning entirely. The demos already contain, for each expert state, a
competent action — so fit the policy to those actions by maximum-likelihood cloning: minimize
$-\mathbb{E}_{(s,a)\sim\text{expert}}[\log\pi(a\mid s)]$ under the Gaussian actor, training both the mean
and the log-std (proper action coverage, not a brittle point regression). No discriminator, no moving
reward, no policy-density to balance — one stationary supervised loss.

**Why it works here.** BC's covariate-shift penalty (compounding $\varepsilon T^2$ drift off the expert
distribution) is blunted because the demos are dense and the experts competent (little extrapolation), and
the *alternatives* are paying a larger adversarial-instability cost. Trading bounded drift for a stable
objective wins when demos are clean.

**Scaffold reality.** This is NOT the classical road-follower clone — no population-coded output, no
perspective-transform recovery synthesis, no pure-pursuit relabeling (the harness exposes none of that).
It uses the one thing the harness *does* expose: the policy's full Gaussian. (1) `RewardNetwork` is a
**dummy** (returns zeros) — BC learns no reward; `compute_reward` returns zeros, so the fixed
reward-normalization and PPO update run on a constant signal and inject no non-stationarity. (2)
`set_policy(policy, optimizer)` is stored *with the optimizer* (unlike AIRL) because BC, not the fixed PPO
step, updates the policy. (3) `update()` runs `n_bc_steps=20` NLL gradient steps per call so cloning
dominates the secondary PPO, with a tiny anti-collapse entropy term ($-0.001\cdot$entropy) and grad-norm
clip 0.5.

**What to watch.** Expect BC strongest overall — winning Hopper and Walker2d decisively (consistently-alive
cloned gaits vs AIRL's two-dead-one-alive) while roughly tying or modestly trailing AIRL on the
non-terminal HalfCheetah, possibly with a wide seed spread (a runaway fast clone, an underfit gait). The
simplest method beating the principled ones because the principle's instability cost outweighs its
occupancy-matching benefit on clean demos.

```python
class RewardNetwork(nn.Module):
    """Dummy reward network for BC (not used for reward shaping).

    BC does not learn a reward; this returns a constant so the PPO
    loop runs but does not meaningfully update from reward signal.
    The policy is trained via supervised loss in IRLAlgorithm.update().
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Unused parameters to keep interface consistent
        self.dummy = nn.Linear(1, 1)

    def forward(self, state, action, next_state):
        return torch.zeros(state.shape[0], device=state.device)


class IRLAlgorithm:
    """BC — Behavioral Cloning.

    Directly trains the policy network to mimic expert actions via
    supervised MSE loss. The reward network is unused.
    Policy is trained both via BC loss in update() and via PPO in the
    main loop (with near-zero reward), but BC dominates learning.
    """

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.total_updates = 0
        # BC does not need a reward optimizer; policy is trained externally
        # We store a reference to policy that gets set during training
        self._policy = None
        self._policy_optimizer = None

    def set_policy(self, policy, optimizer):
        """Set reference to policy for BC updates."""
        self._policy = policy
        self._policy_optimizer = optimizer

    def compute_reward(self, obs, acts, next_obs):
        """BC uses constant reward (PPO loop is secondary)."""
        return torch.zeros(obs.shape[0], device=self.device)

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        """BC supervised update: minimize negative log-probability of expert actions.

        Uses the full policy distribution (mean + log_std) to compute log prob,
        matching the reference imitation library approach. This trains both the
        mean and the variance of the policy, giving better action coverage.
        """
        self.total_updates += 1

        if self._policy is None:
            return {"irl_loss": 0.0}

        batch_size = self.args.irl_batch_size

        # Sample expert data
        n_expert = len(self.expert_demos["obs"])

        total_bc_loss = 0.0
        n_bc_steps = 20  # more BC gradient steps per IRL update

        for _ in range(n_bc_steps):
            expert_idx = torch.randint(0, n_expert, (batch_size,))
            expert_obs = self.expert_demos["obs"][expert_idx]
            expert_acts = self.expert_demos["acts"][expert_idx]

            # Use get_action_and_value to get log_prob of expert actions
            # This trains both actor_mean and actor_logstd
            _, log_prob, entropy, _ = self._policy.get_action_and_value(
                expert_obs, expert_acts,
            )

            # Negative log-likelihood loss (matching reference BC)
            neglogp = -log_prob.mean()
            # Entropy bonus for exploration (prevents policy from collapsing)
            ent_bonus = -0.001 * entropy.mean()

            bc_loss = neglogp + ent_bonus

            self._policy_optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
            self._policy_optimizer.step()
            total_bc_loss += bc_loss.item()

        return {"irl_loss": total_bc_loss / n_bc_steps}
```
