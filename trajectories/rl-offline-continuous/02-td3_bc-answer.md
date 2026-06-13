**Problem.** IQL was *too hedged*: by refusing to ever ascend the critic it stayed pinned near the data
(HalfCheetah 48.1, std 0.12) and underperformed on the stitching task (Maze2d 33.7). I want the opposite
stance — a policy that actually exploits the critic by deterministic policy gradient — with the *minimum*
behavior regularization needed to stop OOD overestimation, because offline every extra knob is untunable.

**Key idea (TD3+BC).** Take stabilized TD3 — twin critics with a `min` target, clipped target-policy
smoothing, delayed actor/Polyak updates — and add one offline change: a behavior-cloning L2 term to the
actor loss, so the ascending actor is anchored to dataset actions. Objective: maximize
`λ·Q(s, π(s)) − (π(s) − a)²`. The actor still climbs Q (real improvement, can commit hard to good
actions and stitch), but the L2 pull keeps it from running off to over-valued OOD actions.

**Why it suppresses overestimation.** The `min` of two target critics caps the bootstrap at the safer
(under-estimating) side; smoothing denies the deterministic actor sharp spurious critic peaks; delay +
soft targets stop the bootstrap chasing a moving estimate. The BC term then anchors the *one* remaining
OOD risk — the actor proposing unseen actions — without a generative behavior model. λ is a *normalizer*,
`λ = α / mean|Q|` (detached), so the RL/BC balance is decoupled from reward scale and one `α = 2.5` works
across datasets.

**Hyperparameters.** `alpha = 2.5`, `policy_noise = 0.2·max_action`, `noise_clip = 0.5·max_action`,
`policy_freq = 2`; twin 2×256 critics (no LayerNorm), 2×256 deterministic actor; clipped double-Q,
Polyak `tau = 5e-3`, `discount = 0.99`, Adam `3e-4`. State normalization from the fixed loop; no reward
preprocessing. Actor gradient ascends `critic_1`.

```python
# EDITABLE region of custom.py — step 2: TD3+BC
class DeterministicActor(nn.Module):
    """Deterministic policy pi(s) = tanh(net(s)) * max_action. 2 x 256 MLP."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q-function Q(s, a). 2 x 256 MLP (TD3+BC reference architecture)."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class OfflineAlgorithm:
    """TD3+BC — Twin Delayed DDPG with Behavior Cloning regularization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        replay_buffer=None,
        discount: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        orthogonal_init: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        # TD3+BC hyperparameters
        self.alpha = 2.5
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2

        # Actor (deterministic) + target
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critics + targets
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)

        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        not_done = 1 - dones.squeeze(-1)
        rewards_flat = rewards.squeeze(-1)
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.critic_1_target(next_states, next_action)
            target_q2 = self.critic_2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards_flat + not_done * self.discount * target_q

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic_1(states, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, actions)
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
```
