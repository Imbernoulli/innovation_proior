## Problem

SPOT's deterministic TD3 actor maximizes `Q(s, π(s))`, so its improvement step still queries the critic
at a self-proposed action — which on a fixed batch is the OOD-extrapolation that caused the
hammer-expert collapse. Keep the off-policy critic and the implicit constraint, but make the policy
improvement *never* query `Q` at an action the data did not contain, and drop the brittle separately-fit
behavior model.

## Key idea

Solve the KL-constrained improvement `max_π E_π[A]` s.t. `KL(π‖π_β) ≤ ε` exactly: `π* ∝ π_β·exp(A/λ)`.
Project onto the parametric policy by *forward* KL and importance-sample from the buffer — the `π_β`
factor **cancels**, leaving an advantage-weighted maximum-likelihood actor update
`θ ← argmax E_{(s,a)~β}[log π_θ(a|s)·exp(A/λ)]`. This reweights *logged* actions only, so the constraint
is implicit (no behavior model, no OOD query during improvement). The advantage uses an off-policy
bootstrapped `Q^π` of the current policy (twin-Q, `min` target, Polyak), with `V(s)` estimated by the
critic at a policy-sampled action.

## Why it works

Forward KL + buffer importance sampling is the only direction that removes both the density model and the
OOD-`Q` query (reverse KL needs both). The same advantage-weighted update runs unchanged offline and
online, so `on_online_start` is a no-op — there is no transition schedule to mis-tune. Per-state `Z(s)`
is dropped (estimating it hurts; it only reweights states) in favor of batch normalization of the weights.

## Hyperparameters

Actor: 3×256 MLP, state-independent `log_std` (`nn.Parameter`, clamp `[−20, 2]`), `Normal` + hard clamp
(not `TanhTransform`). Critics: twin 3×256, unsqueezed `(batch, 1)`, separate Adam optimizers.
`awac_lambda = 0.1`, weight `clamp_max(exp(adv/λ), 100)`. `discount = 0.99`, `tau = 5e-3`, actor/critic
lr `3e-4`. No special transition handling.

## Code

```python
class Actor(nn.Module):
    """AWAC GaussianPolicy — 3x256 MLP, state-independent log_std, Normal + clamp."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 min_log_std: float = -20.0, max_log_std: float = 2.0):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std

    def _get_policy(self, state: torch.Tensor):
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        return torch.distributions.Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        return policy.log_prob(action).sum(-1, keepdim=True)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(-1.0, 1.0)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        return action_t[0].cpu().numpy()


class Critic(nn.Module):
    """Q-function Q(s, a). 3x256 MLP, returns (batch, 1)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([state, action], dim=-1))


class OfflineOnlineAlgorithm:
    """AWAC — Advantage Weighted Actor-Critic for offline-to-online RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        self.awac_lambda = 0.1
        self.exp_adv_max = 100.0

        self.actor = Actor(state_dim, action_dim, 256).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = Critic(state_dim, action_dim, 256).to(device)
        self.critic_2 = Critic(state_dim, action_dim, 256).to(device)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            next_actions, _ = self.actor(next_states)
            q_next = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self.discount * (1.0 - dones) * q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        with torch.no_grad():
            pi_action, _ = self.actor(states)
            v = torch.min(
                self.critic_1(states, pi_action),
                self.critic_2(states, pi_action),
            )
            q = torch.min(
                self.critic_1(states, actions),
                self.critic_2(states, actions),
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self.awac_lambda), self.exp_adv_max
            )

        action_log_prob = self.actor.log_prob(states, actions)
        actor_loss = (-action_log_prob * weights).mean()
        log_dict["actor_loss"] = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, self.device)

    def on_online_start(self):
        # AWAC needs no special handling at transition
        pass
```
