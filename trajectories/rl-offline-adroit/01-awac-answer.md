**Problem.** Offline RL on narrow Adroit `human-v1` data (~25 human trajectories, 24-30 dim actions).
Naive off-policy actor-critic improves the policy by querying Q at policy-proposed actions and
bootstraps from `a' ~ π(s')` — both off the thin demo tube, where the critic extrapolates upward and
the policy chases the error. Behavior cloning is safe but capped at the demonstrators. The goal is some
value-based improvement over the demos while staying inside their support, without fitting a behavior
model (hard on so few narrow trajectories).

**Key idea.** Solve the KL-constrained improvement problem exactly and project its solution onto the
parametric policy by **forward** KL, which lets the behavior-policy factor cancel:

  `max_π E_{a~π}[A(s,a)]  s.t.  KL(π‖π_β) ≤ ε`  ⟹  `π*(a|s) ∝ π_β(a|s)·exp(A(s,a)/λ)`,

and projecting by forward KL with importance sampling from the buffer gives the **advantage-weighted
maximum-likelihood** actor update

  `θ ← argmax_θ E_{(s,a)~buffer}[ exp(A(s,a)/λ) · log π_θ(a|s) ]`,

supervised learning on the dataset's own actions only — so the policy stays in-support implicitly, with
no behavior model and no OOD-Q query during improvement. The advantage uses an off-policy bootstrapped
`Q^π` of the current policy (twin-Q, `min` target, Polyak); `V(s)` is estimated at a policy sample,
`A = Q(s,a) - min_i Q_i(s, a_π)`.

**Why it fits this task.** No behavior model to fit on 25 narrow trajectories; the actor never steps off
the demo tube during improvement. But this harness is **purely offline** (1e6 steps, no online phase —
the method's online-transfer strength is unused), batch and width are fixed at 256, and the manipulation
temperature `λ=0.1` is a sharp, high-variance reweighting on so little data.

**Hyperparameters (this scaffold).** `awac_lambda = 0.1`; `exp_adv_max = 100`; `discount = 0.99`,
`tau = 5e-3`, `actor_lr = critic_lr = 3e-4`; `CONFIG_OVERRIDES = {"normalize": True}`. Actor: 3×256 MLP,
state-independent `log_std` (clamp `[-20, 2]`), **plain Normal** (no TanhTransform), sample clamped to
`[-1,1]`. Critic: 3×256 MLP, output **not** squeezed (`(batch, 1)`); separate optimizer per critic.

```python
# EDITABLE region of custom_adroit.py — step 1: AWAC
CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": True}


class Actor(nn.Module):
    """AWAC GaussianPolicy — 3x256 MLP, state-independent log_std, Normal + clamp."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 min_log_std: float = -20.0, max_log_std: float = 2.0):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([state, action], dim=-1))


class OfflineAlgorithm:
    """AWAC — Advantage Weighted Actor-Critic for offline RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, orthogonal_init=True, device="cuda"):
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

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        # Critic update
        with torch.no_grad():
            next_actions, _ = self.actor(next_states)
            q_next = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self.discount * (1.0 - dones) * q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Actor update (advantage-weighted)
        with torch.no_grad():
            pi_action, _ = self.actor(states)
            v = torch.min(self.critic_1(states, pi_action), self.critic_2(states, pi_action))
            q = torch.min(self.critic_1(states, actions), self.critic_2(states, actions))
            adv = q - v
            weights = torch.clamp_max(torch.exp(adv / self.awac_lambda), self.exp_adv_max)

        action_log_prob = self.actor.log_prob(states, actions)
        actor_loss = (-action_log_prob * weights).mean()
        log_dict["actor_loss"] = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        return log_dict
```
