**Problem.** The advantage-weighted rung put the in-support constraint only in a *stochastic* actor and
read improvement off a single-sample `V(s)`; on narrow Adroit data that was too noisy (Pen swung
32.8↔106.7 across seeds) and the constraint did nothing for the critic's *bootstrap*, which can still
trust an off-support next action (Hammer floored at ~1.0). The goal: move the behavior regularization
into the target construction with a deterministic, low-variance signal, and per-task conservativeness.

**Key idea.** TD3 (twin critics + `min` target, target-policy smoothing, delayed updates) supplies the
overestimation controls; a behavior-cloning penalty is added in **both** places the behavior-regularized
framework allows, instantiated for a deterministic policy as a cheap squared distance — no behavior
model:
- **Actor (policy regularization):** `λ·Q(s,π(s)) - β_actor·Σ(π(s)-a)²`, with `λ = 1/mean(|Q|)`
  (stop-grad) so the RL-vs-imitation balance transfers across reward scales.
- **Critic (value penalty):** pessimize the bootstrap toward the dataset's own recorded next action `â'`
  (provided in the batch), `q ← min_i Q̄_i(s', a') - β_critic·Σ(a' - â')²`, then
  `y = r + γ(1-done)·q`.

The two coefficients are **decoupled** and set per environment, because the acting policy's caution and
the bootstrap's distrust are different jobs and the three Adroit datasets differ in support. LayerNorm
in the **critic** caps `|Q| ≤ ‖w_head‖` on off-support actions; none in the actor.

**Why it fits this task.** Deterministic policy removes the sampled-`V` variance that destabilized the
previous rung; the critic-side penalty closes the downstream OOD-bootstrap leak the actor penalty left
open; per-task `(β_actor, β_critic)` lets the tight Pen tube, the long Hammer sequence and the broad
`door-cloned` mixture each get their own conservativeness; LayerNorm is free against the 256-width
parameter budget.

**Hyperparameters (this scaffold).** Per-env BC coefficients `(β_actor, β_critic)`: Pen `(0.1, 0.5)`,
Hammer `(0.01, 0.5)`, Door `(0.01, 0.1)` (`door-cloned`). `policy_noise = 0.2`, `noise_clip = 0.5`,
`policy_freq = 2`, `normalize_q = True`; `discount = 0.99`, `tau = 5e-3`, `lr = 3e-4`. Deterministic
actor 3×256, no LayerNorm; critic 3×256 with post-activation LayerNorm; CORL-style init. No state
normalization, no MC return floor.

```python
# EDITABLE region of custom_adroit.py — step 2: ReBRAC
# DeterministicActor body replaced (3x256, no LayerNorm, CORL-style init):
class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        import math
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                fan_in = layer.in_features
                if i < len(self.net) - 2:  # hidden layers
                    bound = math.sqrt(1.0 / fan_in)
                    nn.init.uniform_(layer.weight, -bound, bound)
                    nn.init.constant_(layer.bias, 0.1)
                else:  # output layer
                    nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                    nn.init.uniform_(layer.bias, -1e-3, 1e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q with post-activation LayerNorm (ReBRAC critic_ln=True). 3x256 MLP."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )
        import math
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                fan_in = layer.in_features
                if i < len(self.net) - 1:  # hidden layers
                    bound = math.sqrt(1.0 / fan_in)
                    nn.init.uniform_(layer.weight, -bound, bound)
                    nn.init.constant_(layer.bias, 0.1)
                else:  # output layer
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    nn.init.uniform_(layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class OfflineAlgorithm:
    """ReBRAC — TD3+BC with critic BC regularization in the Bellman target."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, orthogonal_init=True, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        env_name = os.environ.get("ENV", "")
        if "hammer" in env_name:
            self.actor_bc_coef = 0.01
            self.critic_bc_coef = 0.5
        elif "door-cloned" in env_name:
            self.actor_bc_coef = 0.01
            self.critic_bc_coef = 0.1
        elif "door" in env_name:
            self.actor_bc_coef = 0.1
            self.critic_bc_coef = 0.1
        else:  # pen (default)
            self.actor_bc_coef = 0.1
            self.critic_bc_coef = 0.5
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.normalize_q = True

        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions_data = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        log_dict: Dict[str, float] = {}

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions_policy = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            bc_penalty = ((next_actions_policy - next_actions_data) ** 2).sum(-1)
            target_q1 = self.critic_1_target(next_states, next_actions_policy)
            target_q2 = self.critic_2_target(next_states, next_actions_policy)
            next_q = torch.min(target_q1, target_q2)
            next_q = next_q - self.critic_bc_coef * bc_penalty
            target_q = rewards + (1.0 - dones) * self.discount * next_q

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            bc_penalty_actor = ((pi - actions) ** 2).sum(-1)
            q_values = torch.min(self.critic_1(states, pi), self.critic_2(states, pi))

            lmbda = 1.0
            if self.normalize_q:
                lmbda = 1.0 / q_values.abs().mean().detach()

            actor_loss = (self.actor_bc_coef * bc_penalty_actor - lmbda * q_values).mean()
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
```
