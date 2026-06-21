## Research question

Offline RL for high-dimensional dexterous manipulation from narrow human demonstrations. The domain is the Adroit family — a 24-DoF simulated hand on Pen (rotation), Door (opening), and Hammer (nailing) — using the D4RL `human-v1` datasets, which hold roughly 25 human teleoperation trajectories per task. The design target is the offline learning algorithm itself: the loss, target construction, policy-extraction rule, and regularization. The surrounding harness — preprocessing, replay buffer, evaluation loop, network widths — is fixed. The core difficulty is that the data forms a thin, expert-but-narrow tube in a 24-to-30-dimensional action space, so value-based methods extrapolate badly once the policy leaves that tube, while pure imitation is starved by the tiny dataset.

## Prior art / Background / Baselines

These are the methods in circulation; the fixed substrate below is the harness they all share.

- **DDPG / TD3.** Deterministic actor-critic with a bootstrapped target, plus clipped double-Q, target-policy smoothing, and delayed actor updates to fight overestimation. Gap: offline, the actor leaves the data tube and the critic's unconstrained off-support estimates drive divergent backups.
- **SAC.** Maximum-entropy off-policy actor-critic with a stochastic Tanh-Gaussian policy and twin critics. Gap: the entropy term and the bootstrap over next actions both query out-of-distribution actions that the static dataset cannot correct.
- **BCQ / BEAR / BRAC.** Offline fixes that constrain the learned policy toward the behavior policy, either by sampling from a learned action generative model (BCQ), an MMD/KL penalty against a fitted behavior model (BEAR), or behavior-regularized actor and critic penalties (BRAC). Gap: they rely on an explicit behavior model that is hard to fit on ~25 narrow human trajectories, and still evaluate Q at sampled actions that may lie off the data support.
- **Behavior cloning.** Regress the policy onto the logged action. It cannot exceed the demonstrations and gives up stitching, but on `human-v1` it is a serious competitor because the data is near-expert.

## Fixed substrate / Code framework

A single offline harness (`custom_adroit.py`) is frozen. It loads the D4RL `human-v1` dataset through a converter that preserves the dataset's own next action `a'` (so the batch is `(s, a, r, s', done, a')`), builds a `ReplayBuffer`, optionally normalizes states by dataset mean/std, and runs `max_timesteps = 1e6` gradient steps at `batch_size = 256`, evaluating every `5e3` steps over 10 rollouts and reporting the D4RL normalized score (`0` = random, `100` = expert). The loop calls `trainer.train(batch)` every step and `eval_actor(env, trainer.actor, ...)` at eval time, so the algorithm must expose `self.actor` with an `.act(state, device)` method. All hidden widths are fixed at 256 (a `_mlp()` factory and a `_max_param_budget()` check guard it), so the contribution must be algorithmic — loss, target, regularization, training procedure — never capacity. Defaults: `discount = 0.99`, `tau = 5e-3`, `actor_lr = critic_lr = 3e-4`.

## Editable interface

Only one region of `custom_adroit.py` (lines 214–416) is editable: a top-of-region `CONFIG_OVERRIDES` dict, the network classes (`DeterministicActor`, `Actor`, `Critic`, `ValueFunction`), and the `OfflineAlgorithm` class (`__init__` builds the nets/optimizers; `train(batch)` does one update and returns a scalar log dict). The harness provides `soft_update`, `init_module_weights`, the `_mlp(in, out, hidden=256, n_layers=3)` factory, and the full dataset via `replay_buffer._states[:size]` etc.

The starting fill is a placeholder: networks are built but `train` does nothing (returns zero losses, no learning). Your approach replaces the network definitions it needs and the body of `OfflineAlgorithm`, and nothing else.

```python
# EDITABLE region of custom_adroit.py — default fill (placeholder, no learning)
CONFIG_OVERRIDES: Dict[str, Any] = {}


class DeterministicActor(nn.Module):
    """Deterministic policy pi(s) = tanh(net(s)) * max_action. 3 x 256 MLP."""

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


class Actor(nn.Module):
    """Tanh-Gaussian stochastic policy. 3 x 256 MLP. For CQL/IQL/AWAC-style methods."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )
        init_module_weights(self.net, orthogonal_init)
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def _get_dist(self, state: torch.Tensor):
        out = self.net(state)
        mean, log_std = torch.split(out, self.action_dim, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return TransformedDistribution(
            Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
        ), mean

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        dist, mean = self._get_dist(state)
        action = torch.tanh(mean) if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return self.max_action * action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist, _ = self._get_dist(state)
        action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
        return dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q(s, a). 3 x 256 MLP."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        init_module_weights(self.net, orthogonal_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    """State value V(s). 3 x 256 MLP. For IQL-style methods."""

    def __init__(self, state_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        init_module_weights(self.net, orthogonal_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class OfflineAlgorithm:
    """Offline RL algorithm — implement your approach here. Default body does nothing."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, orthogonal_init=True, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0
        self.replay_buffer = replay_buffer
        self.actor = Actor(state_dim, action_dim, max_action, orthogonal_init).to(device)
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions = batch
        # Placeholder: replace with your algorithm.
        return {"actor_loss": 0.0, "critic_loss": 0.0}
```

## Evaluation settings

Three Adroit tasks — Pen (`pen-human-v1`), Hammer (`hammer-human-v1`), and Door (`door-cloned-v1`, hidden) — each over three seeds {42, 123, 456}. Metric: D4RL normalized score per task (`0` = random, `100` = expert), averaged over evaluation rollouts. A method is judged on working across the manipulation tasks rather than overfitting one dataset.
