**Problem.** Continuous control off-policy: value-based control needs `argmax_a Q(s,a)`, which has no
cheap solution for a real-valued action and explodes under discretization. The floor must be the
simplest off-policy actor-critic that fills the scaffold's contract — `select_action` for collection,
`update` for one gradient step, `self.actor` exposing `get_action`.

**Key idea (the floor).** Split the value function's two jobs: keep a critic `Q(s,a)` for evaluation
and add a deterministic actor `μ(s)` for the action, moved by the deterministic policy gradient
`∇_θ Q(s,μ_θ(s)) = ∇_θ μ_θ(s)·∇_a Q(s,a)|_{a=μ(s)}`, which amortizes the per-step `argmax`. Because the
target action is just `μ(s')`, off-policy replay needs no action-space importance ratio. The critic
regresses to the deterministic Bellman target `y = r + γ(1−d) Q_targ(s', μ_targ(s'))`; the actor loss
is `−Q(s,μ(s)).mean()`. Self-reference is tamed by slow target copies of *both* actor and critic via
`soft_update` (`τ=0.005`). A single critic, Gaussian exploration noise.

**Why.** Deterministic actor removes the `max`; no action integral in the gradient makes off-policy
replay cheap; target networks keep the bootstrap from chasing itself; the deterministic policy
explores nothing, so behavior noise is added externally — only changing which transitions enter the
buffer, not the learned policy.

**Scaffold edit.** Keep the default deterministic-tanh `Actor` and single `QNetwork`. Fill
`OffPolicyAlgorithm`: add `target_actor`/`qf1_target`, Gaussian-noise `select_action`, and an
`update` that steps the critic every call and the actor + soft-updates every `policy_frequency=2`.

**Hyperparameters.** `γ=0.99`, `τ=0.005`, Adam `3e-4`, batch 256, `exploration_noise=0.1`,
`policy_frequency=2`, single critic. Fixed 256-wide networks (parameter-count enforced).

**What to watch.** A single critic cannot cap overestimation: the actor ascends wherever the critic
bulges upward and reads targets off single sharp points, so expect learning but with value drift and
seed variance that widens on the higher-dimensional hidden Ant. That overestimation is what forces
rung 2.

```python
class OffPolicyAlgorithm:
    """DDPG — Deep Deterministic Policy Gradient."""

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.exploration_noise = args.exploration_noise
        self.policy_frequency = args.policy_frequency
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=args.learning_rate)

    def select_action(self, obs):
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.exploration_noise, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            target_q = self.qf1_target(next_obs, next_actions).view(-1)
            td_target = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.qf1(obs, actions).view(-1)
        critic_loss = F.mse_loss(current_q, td_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            actor_loss = -self.qf1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.qf1_target, self.qf1, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}
```
