## Problem

AWAC's critic only evaluates `Q^{π_β}` and its improvement is one reweighting step from the behavior
policy, so on the heavily-noisy hammer-cloned mixture it cannot find the rare good behavior (collapsed to
0.34). Put genuine multi-step dynamic programming — the ability to *stitch* good fragments across noisy
trajectories — into the value function itself, while still never querying an unseen action.

## Key idea

Read `Q(s,a)` over behavior actions as a per-state random variable: SARSA's MSE gives its mean (what AWAC
had), the in-support max is what improvement needs. Estimate that upper tail with **expectile regression**
(`τ → 1` approaches the support's supremum) using dataset actions only. Split it across two nets so we are
not optimistic about lucky transitions: `V(s)` takes the upper expectile over **actions** (transition
fixed), and `Q` is backed up onto `r + γ V(s')` by honest **MSE** over **dynamics**. Extract the policy by
advantage-weighted regression `exp(β(Q − V))·log π(a|s)` over dataset actions — the same OOD-safe update as
AWAC, now fed a far stronger advantage from a critic that did in-support DP.

## Why it works

`V_τ` is monotone in `τ`, bounded by the in-support optimum, and converges to it as `τ → 1` — spanning
SARSA (`τ = 0.5`) to in-support Q-learning. The higher-`τ` backup propagates value backward across
transitions from different trajectories, which is the stitching hammer-cloned needs. Value training is
policy-free and queries no OOD action; the same update runs offline and online, so `on_online_start` is a
no-op (no transition schedule to mis-tune). Actor dropout and a cosine actor-LR schedule stabilize on the
noisy `cloned` data.

## Hyperparameters

`iql_tau = 0.8`, `beta = 3.0`, `exp_adv_max = 100`. Actor: 2×256 GaussianPolicy, Tanh output,
state-independent `log_std`, dropout 0.1, `Normal` (not `TanhTransform`), CosineAnnealingLR over 1M steps.
Critic: `TwinQ` (two 2×256, squeezed). Value: 2×256 squeezed. `discount = 0.99`, Polyak `tau = 5e-3`,
actor/critic lr `3e-4`. Update order per step: V → Q → Polyak → policy.

## Code

```python
def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Actor(nn.Module):
    """IQL GaussianPolicy — 2x256 MLP with Tanh output, state-independent log_std, Normal dist."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dim: int = 256, n_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        dims = [state_dim] + [hidden_dim] * n_hidden + [action_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.max_action = max_action
        self._log_std_min = -20.0
        self._log_std_max = 2.0

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(self._log_std_min, self._log_std_max))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class TwinQ(nn.Module):
    """Twin Q-functions Q1(s,a), Q2(s,a). 2x256 MLPs, squeezed output."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim] + [hidden_dim] * n_hidden + [1]

        def _build_mlp():
            layers = []
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            return nn.Sequential(*layers)

        self.q1 = _build_mlp()
        self.q2 = _build_mlp()

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    """State value function V(s). 2x256 MLP, squeezed output."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim] + [hidden_dim] * n_hidden + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.v = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state).squeeze(-1)


class OfflineOnlineAlgorithm:
    """IQL — Implicit Q-Learning for offline-to-online RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        self.iql_tau = 0.8
        self.beta = 3.0
        self.exp_adv_max = 100.0

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=int(1e6)
        )

        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.q_target = deepcopy(self.qf).requires_grad_(False).to(device)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=critic_lr)

        self.vf = ValueFunction(state_dim).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            target_q = self.q_target(states, actions)
        v = self.vf(states)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        with torch.no_grad():
            next_v = self.vf(next_states)
        targets = rewards + (1.0 - dones) * self.discount * next_v.detach()
        qs = self.qf.both(states, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
        policy_out = self.actor(states)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, self.device)

    def on_online_start(self):
        # IQL needs no special handling at the offline-to-online transition
        pass
```
