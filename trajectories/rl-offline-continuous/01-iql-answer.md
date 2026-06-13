**Problem.** Offline continuous control: learn from a fixed D4RL dataset with no environment
interaction, without the OOD-action overestimation that diverges naive bootstrapped Q-learning. I want
the *cleanest* floor — value learning that never queries the critic at an action outside the data — so
later rungs can be credited against it.

**Key idea (IQL — implicit Q-learning).** Read Q over the behavior actions at a state as a random
variable. SARSA's MSE gives its mean (= value of π_β, no improvement); the in-support *max* gives true
multi-step DP but needs OOD queries. The τ-expectile — the asymmetric L2 loss |τ − 1(u<0)|·u² — climbs
into the upper tail of that random variable using only dataset actions, reaching the in-support max as
τ → 1. Split the two randomness sources: a value net V(s) takes the upper expectile over *actions*
against a clipped-double-Q target critic (optimistic action selection); Q is then backed up onto
r + γV(s') by *honest MSE* over the transition. No policy and no OOD action ever enter value training.

**Why it suppresses overestimation.** Every Q query in value learning is at a dataset (s, a) pair; the
only "max" is an expectile estimated in-sample, so the critic is never asked to value an unseen action.
The policy is extracted by advantage-weighted regression exp(β(Q − V))·(−log π(a|s)) — reweighting
*observed* actions, never argmax or ∇_aQ ascent — which keeps an implicit stay-near-π_β constraint and
again touches no OOD action.

**Hyperparameters.** `iql_tau = 0.7`, `beta = 3.0`, `EXP_ADV_MAX = 100`; clipped double-Q with Polyak
`tau = 5e-3`, `discount = 0.99`, Adam `3e-4`, cosine-anneal the actor LR over 1e6 steps. 2×256 critic
and value nets; state-independent-log-std Gaussian actor (Tanh-mean MLP). No reward preprocessing.
Update order per step: V (expectile) → Q (MSE) → policy (AWR) → Polyak.

```python
# EDITABLE region of custom.py — step 1: IQL
from torch.optim.lr_scheduler import CosineAnnealingLR

EXP_ADV_MAX = 100.0

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Actor(nn.Module):
    """GaussianPolicy for IQL — state-independent log_std, forward returns Normal."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def forward(self, state: torch.Tensor) -> Normal:
        mean = self.net(state)
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q-function Q(s, a). 2 x 256 MLP (IQL reference architecture)."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    """State value function V(s). 2 x 256 MLP (IQL reference architecture)."""

    def __init__(self, state_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class OfflineAlgorithm:
    """IQL — Implicit Q-Learning with expectile regression and advantage-weighted actor."""

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

        # IQL hyperparameters
        self.beta = 3.0
        self.iql_tau = 0.7

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(1e6))

        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1).requires_grad_(False).to(device)
        self.critic_2_target = deepcopy(self.critic_2).requires_grad_(False).to(device)
        self.q_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_lr,
        )

        self.vf = ValueFunction(state_dim, orthogonal_init).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)

    def _update_v(self, observations, actions, log_dict):
        with torch.no_grad():
            target_q = torch.min(
                self.critic_1_target(observations, actions),
                self.critic_2_target(observations, actions),
            )
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(self, next_v, observations, actions, rewards, dones, log_dict):
        targets = rewards + (1.0 - dones.float()) * self.discount * next_v.detach()
        q1 = self.critic_1(observations, actions)
        q2 = self.critic_2(observations, actions)
        q_loss = (F.mse_loss(q1, targets) + F.mse_loss(q2, targets)) / 2.0
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)

        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict
```
