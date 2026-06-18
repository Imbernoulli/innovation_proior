**Problem.** The decoupled-penalty rung did one-step improvement off a TD3 critic whose target is a
single bootstrapped `min`; it tightened Pen (variance fix) but had no propagated value to climb on the
long Hammer sequence (systematic ~0.35 floor) or the broad `door-cloned` data (~0). The goal: never query
an unseen action in value training, yet still do genuine **multi-step in-support dynamic programming** so
value can stitch across transitions.

**Key idea.** Read `Q(s,a)` over the behavior actions as a per-state random variable: SARSA's MSE gives
its mean (the value of `π_β`, one-step only); the **upper expectile** gives its in-support max — the
improvement operator — estimated from dataset actions only, with no OOD query.
`L_2^τ(u) = |τ - 1(u<0)|·u²` is asymmetric MSE; its τ-expectile → the support's sup as `τ → 1`. To avoid
being optimistic about lucky stochastic transitions, split the estimate:
- **Value (expectile over actions):** `L_V(ψ) = E[L_2^τ(Q̄(s,a) - V_ψ(s))]`, `Q̄ = min(Q1, Q2)`.
- **Q (honest MSE over dynamics):** `L_Q(θ) = E[(r + γ V_ψ(s') - Q_θ(s,a))²]`.

This is provably multi-step DP (monotone in `τ`, → in-support optimum). The policy is extracted by
**advantage-weighted regression** over dataset actions only,
`L_π(φ) = E[-exp(β·(Q̄ - V))·log π_φ(a|s)]` (weight clipped ≤ 100) — improvement with an implicit
constraint, querying nothing unseen.

**Why it fits this task.** Value training touches no off-support action anywhere; expectile DP supplies
the propagated signal the previous rungs lacked on Hammer/Door; AWR is the most reliable extractor on
near-expert data. Dropout in the actor regularizes against memorizing ~25 trajectories.

**Hyperparameters (this scaffold).** `iql_tau = 0.8` (expectile), `beta = 3.0` (AWR temperature),
`exp_adv_max = 100`; `discount = 0.99`, `tau = 5e-3`, `lr = 3e-4`; actor on a `CosineAnnealingLR` over
`1e6` steps; `CONFIG_OVERRIDES = {"normalize": True}`. Actor: **2×256** MLP with **Dropout 0.1** per
hidden layer and **Tanh** mean output, state-independent `log_std`, plain `Normal` (no TanhTransform).
Critic: single twin-Q module (`q1`, `q2`, squeezed). `ValueFunction` for `V`.

```python
# EDITABLE region of custom_adroit.py — step 3: IQL
CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": True}


class Actor(nn.Module):
    """IQL GaussianPolicy — 2x256 MLP with Tanh output + Dropout(0.1),
    state-independent log_std, Normal distribution (no TanhTransform)."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = -20.0
        self._max_log_std = 2.0

    def _get_policy(self, state: torch.Tensor):
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        return Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
        policy = self._get_policy(state)
        return policy.log_prob(action).sum(-1)

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        policy = self._get_policy(state)
        action = policy.mean if deterministic else policy.rsample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = policy.log_prob(action).sum(-1)
        return self.max_action * action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        policy = self._get_policy(state)
        action = policy.sample() if self._mlp.training else policy.mean
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action[0].cpu().numpy()


class Critic(nn.Module):
    """Twin Q-function for IQL. Two 3x256 MLPs, squeeze output to scalar."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def both(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(state, action)
        return torch.min(q1, q2)


class OfflineAlgorithm:
    """IQL — Implicit Q-Learning for offline RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, orthogonal_init=True, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        self.iql_tau = 0.8       # expectile for V loss
        self.beta = 3.0          # inverse temperature for advantage weighting
        self.exp_adv_max = 100.0

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=int(1e6))

        self.qf = Critic(state_dim, action_dim).to(device)
        self.qf_target = deepcopy(self.qf)
        self.qf_target.requires_grad_(False)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=critic_lr)

        self.vf = ValueFunction(state_dim).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)

    def _asymmetric_l2_loss(self, u: torch.Tensor, tau: float) -> torch.Tensor:
        return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        log_dict: Dict[str, float] = {}

        # V update: expectile regression against Q_target
        with torch.no_grad():
            target_q = self.qf_target(states, actions)
        v = self.vf(states)
        adv = target_q - v
        v_loss = self._asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Q update: Bellman with V(s') as bootstrap
        with torch.no_grad():
            next_v = self.vf(next_states)
            q_target = rewards + (1.0 - dones) * self.discount * next_v
        q1, q2 = self.qf.both(states, actions)
        q_loss = (F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)) / 2.0
        log_dict["critic_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.qf_target, self.qf, self.tau)

        # Actor update: advantage-weighted regression
        with torch.no_grad():
            adv_detached = target_q - self.vf(states)
            exp_adv = torch.exp(self.beta * adv_detached).clamp(max=self.exp_adv_max)
        action_log_prob = self.actor.log_prob(states, actions)
        actor_loss = torch.mean(exp_adv * (-action_log_prob))
        log_dict["actor_loss"] = actor_loss.item()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        return log_dict
```
