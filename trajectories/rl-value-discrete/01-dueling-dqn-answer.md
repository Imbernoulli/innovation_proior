**Problem.** Plain DQN puts a single linear head on the fixed features and estimates `|A|` action values
near-independently. In most control states the action barely matters, so those values are nearly equal
and only their common level — the state value `V(s)` — carries information; yet a TD update touches only
the sampled action's output, so `V(s)`, the quantity every bootstrap leans on, is the most diluted thing
in the network.

**Key idea.** Reshape the head, not the algorithm. Split it into a **value stream** `V(s)` (`84 → 1`)
and an **advantage stream** `A(s,·)` (`84 → |A|`) off the fixed encoder, and recombine inside the
forward pass so the module keeps the exact `(batch, |A|)` interface and is driven, unchanged, by the
scaffold's DQN target and squared TD loss. The naive sum `Q = V + A` is unidentifiable (add `c` to `V`,
subtract from `A`, same `Q`), so subtract a per-state reference before forming `Q`. The **max** reference
makes `V` exactly `max_a Q` but jumps when the best action flips; the **mean** reference
`Q = V + (A − mean_{a'} A)` is smooth and only costs the exact semantics (`V` becomes the mean of the
Q-values), so use the mean. Subtracting a per-state constant never changes the action ranking, so the
greedy/epsilon-greedy policy is identical — `V` now sits under all `|A|` outputs and gets a gradient on
every transition.

**Why it should help.** `V` is learned from all the data instead of one action's column; redundant
actions can flatten to ≈0 advantage. Best fit for tasks with long "action doesn't matter" stretches
(LunarLander hovering); little to exploit where every action matters (Acrobot).

**Scaffold-specific choices.** The shared trunk is the **fixed MLP encoder** — no conv torso, so no
`1/√2` trunk-gradient rescale. The target stays the scaffold's **plain DQN max target** (not double
DQN; the loop is frozen). Keep a **global grad-norm clip at 10** as the only optimization change.

**Hyperparameters.** value head `84 → 1`; advantage head `84 → n_actions`; mean aggregator; max-target
squared TD loss; Adam `lr=2.5e-4`; `gamma=0.99`; grad-norm clip `10.0`; everything else the frozen DQN
loop.

```python
# EDITABLE region of custom_value_discrete.py — step 1: Dueling DQN
class QNetwork(nn.Module):
    """Dueling Q-network: MLPEncoder (fixed) + separate value and advantage heads."""

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim)
        # Value stream
        self.value_head = nn.Linear(ENCODER_FEATURE_DIM, 1)
        # Advantage stream
        self.advantage_head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)

    def forward(self, obs):
        features = self.encoder(obs)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ValueAlgorithm:
    """DuelingDQN -- Dueling Deep Q-Network."""

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
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {"td_loss": td_loss.item(), "q_values": old_val.mean().item()}
```
