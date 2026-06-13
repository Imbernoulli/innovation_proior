## Research question

Continuous-control RL from reward alone, off-policy: a simulated body emits a real-valued action
vector each step (joint torques, steering/throttle), and I want a controller learned model-free that
reuses a replay buffer for sample efficiency. The single thing being designed is the off-policy
**actor-critic algorithm** — the losses, the target construction, the exploration, the update rules.
Everything around it (the environment, the replay buffer, the evaluation, the training loop) is
fixed, and the network *dimensions* are fixed: the contribution must be algorithmic, not capacity.

## Prior art before the first rung (value-based control lineage)

The first rung is a deterministic actor-critic. It is the resolution of a line that starts in
value-based control and runs into the wall of continuous actions; these are the methods it reacts to.

- **Deep Q-learning (DQN, Mnih et al. 2015).** Learns `Q(s,a;θ)` with a replay buffer and a target
  network, regressing toward `r + γ max_{a'} Q(s',a';θ⁻)`. Cracked Atari from pixels, but its control
  rule is a greedy `argmax` over actions, and its target is a greedy max too. Gap: the `max_{a'}` is a
  reduction over a short discrete vector — there is no such enumeration for a real-valued action.
- **Discretizing the action space.** Bin each actuator and run DQN over the product set. Gap: the
  count is exponential in the number of actuators (`3^7 = 2187` for seven joints at three levels
  each, before any fine resolution), and the grid throws away the metric structure that makes 0.30
  and 0.31 nearby torques — a dead end.
- **The deterministic policy gradient (Silver et al. 2014).** Remove the `max` by splitting the two
  jobs the value function was doing: keep a critic `Q(s,a)` for evaluation, and add a separate
  differentiable actor `μ(s)` that emits the action. The actor is moved to increase `Q(s,μ(s))` by the
  chain rule `∇_θ Q(s,μ_θ(s)) = ∇_θ μ_θ(s)·∇_a Q(s,a)|_{a=μ(s)}` — amortizing the per-step `argmax`
  into a network. Gap: this is the theory; turning it into a stable deep learner still needs replay,
  target networks, and explicit exploration bolted on, and the coupled actor-critic is delicate.
- **Experience replay + target networks (from DQN, reused here).** The two stabilizers the deep
  version inherits: a FIFO buffer to decorrelate samples and reuse data off-policy, and slowly
  tracked target copies so the bootstrap target does not chase the live network. Gap: a deterministic
  actor explores nothing on its own, so off-policy exploration noise must be added separately.

## The fixed substrate

A single-environment off-policy loop is frozen and must not be touched. It seeds, builds one
Gymnasium MuJoCo env (`HalfCheetah-v4`, `Reacher-v4`, hidden `Ant-v4`), and a `SimpleReplayBuffer`
holding `(obs, next_obs, action, reward, done)`. For `learning_starts` steps it acts uniformly at
random to fill the buffer; afterward it calls `algorithm.select_action(obs)`, steps the env, stores
the transition (with time-limit truncation written into `real_next_obs` so a truncated step still
bootstraps), and once past `learning_starts` samples a minibatch every step and calls
`algorithm.update(batch)`. Periodically it calls `eval_actor(env_id, algorithm.actor, ...)`, which
runs the policy greedily via `algorithm.actor.get_action(obs)` (it unwraps a tuple return, so a
stochastic actor may return `(action, …)`). The loop provides helpers: `soft_update(target, source,
tau)` for Polyak averaging, and `_mlp_factory(in, out, hidden=256)` for a 2-hidden-layer MLP. Config
is fixed: `gamma=0.99`, `tau=0.005`, `batch_size=256`, `learning_rate=3e-4`, `buffer_size=1e6`,
`learning_starts=25000`, `policy_frequency=2`, `exploration_noise=0.1`, `total_timesteps=1e6`.

## The editable interface

Exactly one region is editable: the `Actor`, `QNetwork`, and `OffPolicyAlgorithm` classes in
`custom_offpolicy_continuous.py` (lines 153–244). The contract the fixed loop relies on:
`OffPolicyAlgorithm(obs_dim, action_dim, max_action, device, args)` builds the algorithm and **must**
set `self.actor` to an `nn.Module` with `.get_action(obs)`; `select_action(obs)` returns a numpy
action for data collection (with exploration noise); `update(batch)` does one gradient update on
`batch = (obs, next_obs, actions, rewards, dones)` (tensors on device) and returns a metrics dict.
The network *dimensions* (256-wide) are fixed and a runtime parameter-count check enforces it.

The starting point is the scaffold default: a deterministic tanh actor, a single Q-critic, and an
`update` that does nothing (a placeholder returning zero losses). Each rung replaces exactly these
definitions.

```python
# EDITABLE region of custom_offpolicy_continuous.py (lines 153-244) — default scaffold fill
class Actor(nn.Module):
    """Actor network. forward(obs) -> action (training); get_action(obs) -> action (eval, no grad)."""

    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor(max_action, dtype=torch.float32))

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x)) * self.action_scale

    @torch.no_grad()
    def get_action(self, obs):
        return self.forward(obs)


class QNetwork(nn.Module):
    """Q-function Q(s, a) -> scalar."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OffPolicyAlgorithm:
    """Off-policy actor-critic -- implement the approach here. Must set self.actor."""

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.total_it = 0
        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=args.learning_rate)

    def select_action(self, obs):
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        action = self.actor(obs_t).cpu().numpy().flatten()
        noise = np.random.normal(0, self.max_action * 0.1, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch
        # Placeholder -- replace with the algorithm
        return {"critic_loss": 0.0, "actor_loss": 0.0}
```

## Evaluation settings

Trained for `total_timesteps = 1e6` environment steps and evaluated on Gymnasium MuJoCo
continuous-control tasks: **HalfCheetah-v4** and **Reacher-v4** (reported), with a hidden
**Ant-v4** (higher-dimensional, harder dynamics). Three seeds {42, 123, 456}. Metric: mean episodic
return over the evaluation episodes, higher is better on every environment. A strong method should
transfer across environments with different dynamics and action effects, not just win one.
