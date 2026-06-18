## Research question

Offline RL for high-dimensional dexterous manipulation from *narrow* human-demonstration data. The
target is the Adroit family — a 24-DoF simulated hand on Pen (rotation), Door (opening) and Hammer
(nailing) — trained from the D4RL `human-v1` datasets, which hold only ~25 human teleoperation
trajectories per task. The single thing being designed is the **offline learning algorithm** itself:
the loss, the target construction, the policy-extraction rule, the regularization. Everything around it
— preprocessing, the replay buffer, the evaluation loop, the network widths — is fixed. The signature
difficulty is that the data is a thin, expert-but-narrow tube in a 24-to-30-dimensional action space,
so any value-based method extrapolates badly the instant the policy steps off that tube, while pure
imitation is starved by the tiny dataset.

## Prior art before the first rung (offline-RL lineage)

The first rung reacts to the standard off-policy actor-critic and the offline fixes that grew up around
it. These are the methods the ladder departs from; the fixed substrate below is the harness they all
fill.

- **DDPG / TD3 (Lillicrap et al. 2016; Fujimoto et al. 2018).** Deterministic actor-critic with a
  bootstrapped target `y = r + γ Q̄(s', π(s'))`. TD3 adds clipped double-Q (`min` of twins), target
  policy smoothing (`σ=0.2`, clip `0.5`), and delayed actor/target updates (every 2 steps) to fight
  overestimation. Gap: nothing keeps the policy on the data — offline, the actor walks off the demo
  tube and the critic's unconstrained off-support surface feeds a divergent backup.
- **SAC (Haarnoja et al. 2018).** Maximum-entropy off-policy actor-critic, a stochastic Tanh-Gaussian
  policy with twin critics. Strong online; offline, dropping the dataset into its buffer extracts
  essentially nothing — the entropy-seeking policy and the bootstrap `a' ~ π` both query
  out-of-distribution actions the static dataset can never correct. Gap: no in-distribution constraint.
- **BCQ / BEAR / BRAC (Fujimoto et al. 2019; Kumar et al. 2019; Wu et al. 2019).** The first offline
  fixes: constrain `π` toward the behavior policy `π_β`, either by sampling from a learned generative
  model of dataset actions (BCQ), an MMD/KL penalty against a fitted `π̂_β` (BEAR), or a
  behavior-regularized actor *and* critic penalty (BRAC). They stabilize offline learning, but each
  fits and leans on an explicit behavior model `π̂_β` — fragile on ~25 narrow human trajectories — and
  still, at some point, evaluates a learned `Q` at a sampled, possibly off-support action. Gap: an
  explicit behavior model that is hard to fit on tiny narrow data, and a residual OOD query.
- **Behavior cloning.** The honest floor on this data: regress `π(s)` onto the logged action. It
  cannot exceed the demonstrations and gives up all stitching, but on `human-v1` it is a serious
  competitor precisely because the data is near-expert. The ladder must beat it to justify doing RL.

## The fixed substrate

A single offline harness (`custom_adroit.py`) is frozen. It loads the D4RL `human-v1` dataset through a
ReBRAC-style converter that **preserves the dataset's own next action** `â'` (so the batch is
`(s, a, r, s', done, â')`), builds a `ReplayBuffer`, optionally normalizes states by dataset
mean/std (`CONFIG_OVERRIDES = {"normalize": ...}`), and runs `max_timesteps = 1e6` gradient steps at
`batch_size = 256`, evaluating every `5e3` steps over 10 rollouts and reporting the D4RL normalized
score (`0` = random, `100` = expert). The loop calls `trainer.train(batch)` every step and
`eval_actor(env, trainer.actor, ...)` at eval time, so the algorithm **must** expose `self.actor` with
an `.act(state, device)` method. A hard architectural rule is enforced: **all hidden widths are 256**
(a `_mlp()` factory and a `_max_param_budget()` check guard it), so the contribution has to be
*algorithmic* — loss, target, regularization, training procedure — never capacity. Defaults:
`discount = 0.99`, `tau = 5e-3`, `actor_lr = critic_lr = 3e-4`.

## The editable interface

Exactly one region (`custom_adroit.py` lines 214–416) is editable: a top-of-region `CONFIG_OVERRIDES`
dict, the network classes (`DeterministicActor`, `Actor`, `Critic`, `ValueFunction`), and the
`OfflineAlgorithm` class (`__init__` builds the nets/optimizers; `train(batch)` does one update and
returns a scalar log dict). Every method on the ladder is a fill of this same contract. The harness
provides `soft_update`, `init_module_weights`, the `_mlp(in, out, hidden=256, n_layers=3)` factory, and
the full dataset via `replay_buffer._states[:size]` etc.

The starting point is the scaffold default: networks are built but `train` is a **placeholder that does
nothing** (returns zero losses, no learning). Each method replaces the network definitions it needs and
the body of `OfflineAlgorithm`, and nothing else.

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

Three Adroit tasks — **Pen** (`pen-human-v1`), **Hammer** (`hammer-human-v1`) and **Door**
(`door-cloned-v1`, hidden) — each over three seeds {42, 123, 456}. Metric: the D4RL normalized score
per task (`0` = random, `100` = expert), averaged over evaluation rollouts; higher is better on all
three. A method is judged on working *across* the manipulation tasks rather than overfitting one
dataset.
