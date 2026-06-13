**Problem (from rung 2).** SAC fixed Ant (1243→3577) with twin-min + entropy, but the entropy taxed
the tasks that want a sharp policy: HalfCheetah dropped 11514→10360, Reacher worsened −3.94→−4.76
across every seed. Wanted: the overestimation control of twin critics on a *deterministic* actor, so
there is no entropy tax at evaluation.

**Key idea.** Attack function-approximation error three ways on a deterministic actor-critic:
1. **Clipped double-Q.** Form the target from `min(Q1_targ, Q2_targ)` at the next action. If Q2 ≥ Q1
   it is the standard single-critic target (no added bias); if Q2 < Q1 it is the Double-Q correction.
   Caps the bias toward the safer underestimation side — the part of SAC that actually fixed Ant.
2. **Delayed policy + target updates.** Update the actor and targets only every `d` critic steps
   (`policy_frequency=2`): let the critic settle before the policy moves against it (two timescales).
3. **Target-policy smoothing.** Perturb the target action with clipped Gaussian noise
   `ε∼clip(N(0,σ̃),−c,c)`, clipped back into range, before evaluating the critic — fit a neighborhood,
   not a knife-edge point. This puts back, on a deterministic actor, the target smoothing SAC got for
   free from its stochastic policy — without keeping the policy stochastic at deployment.

**Why.** The actor ascends the critic and chases its overestimates (proven for the gradient-driven
actor, no max needed); twin-min caps it; bootstrapping accumulates error as variance, restrained by a
slow target network, which argues for delaying the actor; the deterministic actor overfits sharp
critic peaks, defused by smoothing. Deterministic at evaluation = no entropy tax on Reacher/HalfCheetah.

**Scaffold edit.** Revert `Actor` to the default deterministic-tanh (returns a plain action) and
`select_action` to deterministic + Gaussian exploration noise. Fill `OffPolicyAlgorithm` with twin
critics + twin target critics + a *target actor*, no entropy machinery, `policy_noise=0.2`,
`noise_clip=0.5`; smoothed clipped twin-min target every step, deterministic actor step + target
soft-updates every `policy_frequency=2`.

**Hyperparameters.** `γ=0.99`, `τ=0.005`, Adam `3e-4`, batch 256, `policy_frequency=2`,
`policy_noise=0.2`, `noise_clip=0.5`, `exploration_noise=0.1`, twin critics. Fixed 256-wide networks.

**What to watch.** Ant should hold near or above SAC (twin-min kept). Reacher should beat SAC's −4.76
back toward −3.94 (no eval entropy). HalfCheetah should claw back above 10360 toward 11514. The win
condition is being strong on all three at once. Residual: still learns through *stale target
networks* — `τ` balances a deliberately-lagged signal — which is what a further rung would attack.

```python
class OffPolicyAlgorithm:
    """TD3 — Twin Delayed Deep Deterministic Policy Gradient."""

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.exploration_noise = args.exploration_noise
        self.policy_frequency = args.policy_frequency
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.qf2 = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf2_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=args.learning_rate,
        )

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
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            ) * self.max_action
            next_actions = (self.target_actor(next_obs) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.qf1_target(next_obs, next_actions).view(-1)
            target_q2 = self.qf2_target(next_obs, next_actions).view(-1)
            td_target = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        q1 = self.qf1(obs, actions).view(-1)
        q2 = self.qf2(obs, actions).view(-1)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

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
            soft_update(self.qf2_target, self.qf2, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}
```
