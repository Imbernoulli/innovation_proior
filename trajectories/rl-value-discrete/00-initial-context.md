## Research question

Design a value-based reinforcement-learning algorithm for **discrete** action spaces and run it on small-to-medium control tasks (CartPole, LunarLander, Acrobot) within a fixed interaction budget. The design target is the *value head and the value-learning update* — how `Q(s,a)` is represented, what bootstrapped target it regresses toward, and what loss performs the regression. Everything else — the encoder, the replay buffer, epsilon-greedy collection, target-network sync, evaluation protocol — is frozen. The single design decision is what sits on top of an 84-dimensional feature vector and how it is trained.

## Prior art / Background / Baselines

The substrate is plain DQN, the current baseline for value-based discrete control.

- **Tabular Q-learning (Watkins 1989).** It learns `Q(s,a)` by stochastic fixed-point iteration toward `r + γ max_{a'} Q(s',a')`, storing one entry per state-action pair.
- **Neural fitted Q / function approximation (Riedmiller 2005).** It replaces the table with a neural-network regressor `Q(s,a;θ)` trained by regression on bootstrapped targets.
- **DQN (Mnih et al. 2015).** It stabilizes neural fitted Q with an experience-replay buffer and a periodically frozen target network that supplies the bootstrap target, with epsilon-greedy action selection from a single linear head.

## Fixed substrate / Code framework

A single-environment off-policy value-learning loop is frozen and must not be touched. It maintains a numpy replay buffer of one-step transitions `(obs, next_obs, action, reward, done)`, samples **uniformly** with `batch_size=128`, uses a linear epsilon schedule (`start_e=1 → end_e=0.05` over the first half of training), a `learning_starts=10000` warm-up, training every `train_frequency=10` steps, a **hard** target-network sync every `target_network_frequency=500` steps (`tau=1.0`), Adam at `learning_rate=2.5e-4`, `gamma=0.99`, `total_timesteps=500000`, and greedy evaluation (`eval_episodes=10`) on a fresh env. The feature extractor is a **fixed** two-layer MLP encoder `obs_dim → 120 → 84` with ReLUs; its dimensions are locked and a runtime parameter-count check enforces that the contribution is algorithmic, not encoder capacity. The loop calls exactly three methods on the algorithm: `select_action(obs, epsilon)` during collection, `update(batch, global_step)` after each train step, and reads `algorithm.q_network` for evaluation.

## Editable interface

Exactly one region is editable: the `QNetwork` class (the head on top of `MLPEncoder`) and the `ValueAlgorithm` class (`__init__`, `select_action`, `update`) in `custom_value_discrete.py`. `QNetwork.forward(obs)` must return per-action `Q`-values `(batch, n_actions)`; `ValueAlgorithm.update` receives the uniform replay batch and returns a dict of scalar metrics. The replay buffer, encoder, epsilon schedule, target-sync cadence, and loop are fixed — so n-step returns and prioritized replay are out of reach; only the head, the target, and the loss can change.

The starting point is **plain DQN**: a single linear head and the squared one-step TD loss with a max-over-actions target.

```python
# EDITABLE region of custom_value_discrete.py — default fill (plain DQN)
class QNetwork(nn.Module):
    """Q-network: MLPEncoder (fixed) + linear head. forward(obs) -> (batch, n_actions)."""

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim)                       # fixed: obs_dim -> 120 -> 84
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)    # single linear head

    def forward(self, obs):
        features = self.encoder(obs)
        return self.head(features)


class ValueAlgorithm:
    """DQN -- Deep Q-Network with a max-target squared TD loss."""

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions).to(device)
        self.target_network = QNetwork(obs_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        q_values = self.q_network(obs_t)
        return torch.argmax(q_values, dim=1).item()

    def update(self, batch, global_step):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            target_max, _ = self.target_network(next_obs).max(dim=1)
            td_target = rewards + (1 - dones) * self.gamma * target_max

        old_val = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return {"td_loss": td_loss.item(), "q_values": old_val.mean().item()}
```

## Evaluation settings

Three Gymnasium discrete-control tasks span the difficulty range — **CartPole-v1** (easy; return capped at 500), **LunarLander-v2** (harder; dense shaped reward but long-horizon credit assignment and a deceptive crash basin, returns roughly −400 to +300), and **Acrobot-v1** (sparse-ish; return is negative time-to-goal, roughly −500 to −60) — each over three seeds {42, 123, 456}. The metric on every task is the **mean greedy evaluation episodic return** at the fixed budget (higher is better on all three). Per-task elapsed time is logged but is not the objective.
