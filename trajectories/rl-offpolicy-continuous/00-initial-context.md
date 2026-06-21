## Research question

Continuous-control RL from reward alone, off-policy: a simulated body emits a real-valued action vector each step, and the goal is a model-free controller that reuses a replay buffer for sample efficiency. The design object is the off-policy actor-critic algorithm — the losses, target construction, exploration, and update rules. The environment, replay buffer, evaluation, training loop, and network dimensions are fixed; the contribution must be algorithmic.

## Prior art / Background / Baselines

- **Deep Q-learning (DQN).** Learns `Q(s,a)` with a replay buffer and target network, using a greedy max over actions for both control and bootstrap targets. Gap: the max is only defined over a finite discrete action set, so it does not apply to real-valued actions.
- **Discretizing the action space.** Bins each actuator and runs DQN over the product set. Gap: the product grows exponentially with the number of actuators and loses the metric structure between nearby torque values.
- **Deterministic policy gradient (DPG).** Replaces the action max with a differentiable deterministic actor `μ(s)`, obtaining the policy gradient by chaining through a critic `Q(s,a)`. Gap: it is a theoretical gradient theorem; a stable deep off-policy learner for continuous control has not yet been demonstrated, and the coupled actor-critic optimization is fragile.
- **Experience replay + target networks.** Stores past transitions in a FIFO buffer for off-policy reuse and uses slowly-updated target copies for bootstrap targets. Gap: these stabilize discrete-action value learning but do not address continuous actions or the exploration problem of a deterministic actor.

## Fixed substrate / Code framework

A single-environment off-policy loop is frozen. It seeds, builds one Gymnasium MuJoCo env (`HalfCheetah-v4`, `Reacher-v4`, hidden `Ant-v4`), and a `SimpleReplayBuffer` holding `(obs, next_obs, action, reward, done)`. For `learning_starts` steps it acts uniformly at random; afterward it calls `algorithm.select_action(obs)`, steps the env, stores the transition (with time-limit truncation written into `real_next_obs` so a truncated step still bootstraps), and samples a minibatch every step for `algorithm.update(batch)`. Periodically it calls `eval_actor(env_id, algorithm.actor, ...)`, which runs the policy greedily via `algorithm.actor.get_action(obs)` (it unwraps a tuple return, so a stochastic actor may return `(action, …)`). The loop provides `soft_update(target, source, tau)` and `_mlp_factory(in, out, hidden=256)`. Config: `gamma=0.99`, `tau=0.005`, `batch_size=256`, `learning_rate=3e-4`, `buffer_size=1e6`, `learning_starts=25000`, `policy_frequency=2`, `exploration_noise=0.1`, `total_timesteps=1e6`.

## Editable interface

Exactly one region is editable: the `Actor`, `QNetwork`, and `OffPolicyAlgorithm` classes in `custom_offpolicy_continuous.py` (lines 153–244). The contract: `OffPolicyAlgorithm(obs_dim, action_dim, max_action, device, args)` builds the algorithm and must set `self.actor` to an `nn.Module` with `.get_action(obs)`; `select_action(obs)` returns a numpy action for data collection with exploration noise; `update(batch)` performs one gradient update on `batch = (obs, next_obs, actions, rewards, dones)` and returns a metrics dict. Network dimensions (256-wide) are fixed and enforced by a runtime parameter-count check.

The starting point is the scaffold default: a deterministic tanh actor, a single Q-critic, and an `update` that returns zero losses. Each rung replaces exactly these definitions.

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

Trained for `total_timesteps = 1e6` environment steps and evaluated on Gymnasium MuJoCo continuous-control tasks: **HalfCheetah-v4** and **Reacher-v4** (reported), with a hidden **Ant-v4** (higher-dimensional, harder dynamics). Three seeds {42, 123, 456}. Metric: mean episodic return over evaluation episodes, higher is better. A strong method transfers across environments with different dynamics and action effects.
