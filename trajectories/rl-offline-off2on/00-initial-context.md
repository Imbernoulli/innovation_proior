## Research question

Offline-to-online RL pretrains a policy and value function on a fixed dataset `D` and then continues
learning with live environment interaction. The single thing being designed is the
`OfflineOnlineAlgorithm` class — the networks it builds, the per-batch `train` update, and the
`on_online_start` hook fired once at the offline→online handoff. Everything around it (data loading,
the replay buffer, the two-phase training loop, evaluation, the 256-width network cap) is fixed.

The transition is where this problem bites. A conservative offline value function can turn
overoptimistic the moment online data shifts the replay distribution; a behavior-regularized policy can
forget the competence it learned offline; and naive fine-tuning routinely triggers an early
**Q-value collapse** and a performance drop before any recovery. The datasets are Adroit `cloned-v1`
(Pen, Door, Hammer) — mixtures of expert and noisy demonstrations, so offline pretraining never
produces a strong policy on its own and the online phase has to improve substantially **without losing
the little competence that was learned**. The precise question: design one algorithm that pretrains
stably from this mixed offline data and then fine-tunes online without catastrophic forgetting or
Q-collapse.

## Prior art before the first rung (offline-to-online lineage)

The ladder reacts to the standard offline-RL families, each of which pretrains stably but stumbles at
the online handoff in its own way.

- **Off-policy actor-critic on a static batch (TD3 / SAC, Fujimoto et al. 2018; Haarnoja et al. 2018).**
  Twin critics fit a Bellman target `r + γ Q_target(s', π(s'))` and the actor maximizes `Q(s, π(s))`.
  Online this self-corrects; on a fixed batch the bootstrap evaluates the critic at the *actor's* chosen
  `π(s')` — an action the data may never contain — where the net extrapolates upward, the actor (a
  maximizer) is pulled toward it, and the inflated value bootstraps and diverges. Gap: no mechanism to
  keep the actor inside the data's support.
- **Policy-constraint offline RL (BCQ, BEAR, BRAC, TD3+BC; Fujimoto et al. 2019, 2021).** Add a penalty
  or architecture that keeps `π` close to the behavior policy `π_β`, which stabilizes offline learning.
  Gap: most pin the policy to a divergence-from-`π_β` (or to an explicitly fit behavior model), which is
  over-conservative on mixed data and, once online, either strangles improvement or — if the constraint
  is fixed — leaves the early Q-collapse unaddressed.
- **Value-regularization offline RL (CQL; Kumar et al. 2020).** Push `Q` down on out-of-distribution
  actions and up on dataset actions for a conservative lower bound. Strong offline. Gap: the learned
  `Q` is *uncalibrated* in scale — it can sit far below the true return — so at the online transition
  the first real returns look huge relative to it, the critic lurches, and the policy collapses before
  recovering.

The three rungs below are the families adapted to *this* task's harness, weakest to strongest, each a
fill of the same editable contract.

## The fixed substrate

A two-phase loop is frozen and must not be touched. **Phase 1 (offline, 1M gradient steps):** sample
minibatches from a `ReplayBuffer` preloaded with the D4RL `cloned-v1` dataset and call
`trainer.train(batch, is_online=False)`. **Transition:** `trainer.on_online_start()` fires once.
**Phase 2 (online, 1M env-interaction steps):** at each step `trainer.select_action(state)` collects a
transition that is appended to the *same* buffer (so it grows with online data), then
`trainer.train(batch, is_online=True)` runs. States are normalized by dataset mean/std (toggleable via
`CONFIG_OVERRIDES`). The loop provides `soft_update` (Polyak), `init_module_weights`, an `_mlp` factory
locked at hidden width 256, the `ReplayBuffer` (whose `sample` returns
`[states, actions, rewards, next_states, dones, next_actions]`), and `eval_actor`. If the trainer
defines an optional `pretrain(replay_buffer, batch_size)` method it is called once before Phase 1.

Two hard constraints from the harness: **all MLP hidden widths must be 256**, and the **total trainable
parameter count is capped at ~1.2× the largest baseline architecture** — the contribution must be
algorithmic (transition handling, value calibration, replay balancing, constraint annealing), not
capacity. Metric: D4RL normalized score (0 = random, 100 = expert), higher is better, on Pen, Door, and
Hammer `cloned-v1`.

## The editable interface

Exactly one region is editable — the network classes (`DeterministicActor`, `Actor`, `Critic`,
`ValueFunction`) and the `OfflineOnlineAlgorithm` class, plus an optional `CONFIG_OVERRIDES` dict
(allowed keys: `normalize`, `normalize_reward`, `actor_lr`, `critic_lr`, `tau`, `expl_noise`,
`discount`). Every rung replaces this region and nothing else. The contract the loop calls:
`__init__(state_dim, action_dim, max_action, replay_buffer=..., ...)` builds the nets and sets
`self.actor` to a module with `.act(state, device)`; `train(batch, is_online)` runs one update and
returns a scalar-metric dict; `select_action(state)` collects online data (may add exploration noise);
`on_online_start()` adjusts hyperparameters/optimizers at the handoff. The starting point is the
scaffold default below — a plain twin-critic actor-critic stub whose `train` is a no-op placeholder.

```python
# EDITABLE region of CORL/algorithms/finetune/custom_finetune.py — default fill (placeholder)
CONFIG_OVERRIDES: Dict[str, Any] = {}


class DeterministicActor(nn.Module):
    """Deterministic policy pi(s) = tanh(net(s)) * max_action. Default: 2 x 256 MLP."""

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
    """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP."""

    def __init__(self, state_dim, action_dim, max_action, orthogonal_init=False):
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
        self.log_std_min, self.log_std_max = -20.0, 2.0

    def _get_dist(self, state):
        out = self.net(state)
        mean, log_std = torch.split(out, self.action_dim, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return TransformedDistribution(
            Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
        ), mean

    def forward(self, state, deterministic=False):
        dist, mean = self._get_dist(state)
        action = torch.tanh(mean) if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return self.max_action * action, log_prob

    def log_prob(self, state, action):
        dist, _ = self._get_dist(state)
        action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
        return dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q(s, a). Default: 3 x 256 MLP, squeezed output."""

    def __init__(self, state_dim, action_dim, orthogonal_init=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        init_module_weights(self.net, orthogonal_init)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    """V(s). Default: 3 x 256 MLP, squeezed output."""

    def __init__(self, state_dim, orthogonal_init=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        init_module_weights(self.net, orthogonal_init)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class OfflineOnlineAlgorithm:
    """Offline-to-Online RL algorithm — implement your approach here."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0
        self.replay_buffer = replay_buffer

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions = batch
        # ── Placeholder: replace with your algorithm ──
        return {"actor_loss": 0.0, "critic_loss": 0.0}

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, self.device)

    def on_online_start(self):
        pass
```

## Evaluation settings

Each method is trained and evaluated on **Pen**, **Door**, and **Hammer** `cloned-v1` (the public test
commands run pen-cloned-v1 and hammer-cloned-v1; a hammer-expert-v1 run is held out). Per method the
offline phase trains for 1M gradient steps and the online phase for 1M environment-interaction steps,
with periodic evaluation throughout both phases. Three seeds {42, 123, 456}. Metric: D4RL normalized
score, mean over 10 evaluation episodes, higher is better; a strong method retains offline competence
while benefiting from online fine-tuning across all three manipulation tasks.
