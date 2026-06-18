**Problem.** QR-DQN learns `N` equal-mass quantile **locations**, so resolution is spread evenly across
the probability axis and the low-probability crash tail of LunarLander — the rare large-negative outcome
that decides the worst seeds — gets the same one-in-50 mass budget as the dense middle. On a single
500k-step run those tail locations are the noisiest part of the head, leaving LunarLander at a good-not-
great 197 with ragged seeds. The classic-control return range is known and bounded, so fix the locations
and learn the mass.

**Key idea.** A **categorical** return distribution on a *fixed* grid `z_i = v_min + i·Δz`, with the
softmax **probabilities** `p_i(s,a)` learnable: `Z_θ = Σ_i p_i δ_{z_i}`. The distributional Bellman
operator still contracts only in Wasserstein, and sampled Wasserstein has a biased gradient — but a
fixed grid lets me use **cross-entropy**, which is unbiased. The geometric obstruction (`T̂z_j = r + γz_j`
lands off-grid) is fixed by **projecting** the shifted target onto its two nearest atoms by linear
interpolation (mass-preserving, with an integer-`b` correction so mass is not dropped), then minimizing
`−Σ_i m_i log p_i`. The Bellman update becomes multiclass classification.

**Why it should help.** Crash-tail mass is now a softmax probability on a grid point at the tail's actual
return magnitude, learned by stable cross-entropy — honest tail resolution where QR-DQN had one slippery
location. Cost: KL is value-blind and the grid must cover the range; acceptable on bounded classic
control.

**Scaffold-specific choices.** Head = linear `84 → |A|·N` softmaxed over atoms on the **fixed MLP
encoder**; `forward` returns `Σ_i z_i p_i` per action. **`N=51`**; support **`v_min=−500, v_max=+500`**
(not the generic `[−10,10]` — the unclipped classic-control returns span this; `Δz=20`). Greedy on the
mean; target from the target net; terminals via `(1−dones)`. No dueling/quantile head — isolate the
categorical effect.

**Hyperparameters.** `n_atoms=51`; `v_min=−500.0`; `v_max=500.0`; cross-entropy loss after the
linear-interpolation projection; Adam `lr=2.5e-4`; `gamma=0.99`; everything else the frozen DQN loop.

```python
# EDITABLE region of custom_value_discrete.py — step 3: C51
class QNetwork(nn.Module):
    """Distributional Q-network for C51: MLPEncoder (fixed) + n_actions x n_atoms head."""

    def __init__(self, obs_dim, n_actions, n_atoms=51, v_min=-500, v_max=500):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.encoder = MLPEncoder(obs_dim)
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions * n_atoms)

    def forward(self, obs):
        features = self.encoder(obs)
        logits = self.head(features)
        pmfs = torch.softmax(logits.view(len(obs), self.n_actions, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        return q_values

    def get_action(self, obs, action=None):
        features = self.encoder(obs)
        logits = self.head(features)
        pmfs = torch.softmax(logits.view(len(obs), self.n_actions, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(obs)), action]


class ValueAlgorithm:
    """C51 -- Categorical DQN with distributional value learning."""

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.n_atoms = 51
        self.v_min = -500.0
        self.v_max = 500.0
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions, self.n_atoms, self.v_min, self.v_max).to(device)
        self.target_network = QNetwork(obs_dim, n_actions, self.n_atoms, self.v_min, self.v_max).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        action, _ = self.q_network.get_action(obs_t)
        return action.item()

    def update(self, batch, global_step):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            _, next_pmfs = self.target_network.get_action(next_obs)
            next_atoms = rewards.unsqueeze(1) + self.gamma * self.target_network.atoms * (1 - dones.unsqueeze(1))
            # Projection
            delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
            tz = next_atoms.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / delta_z
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.q_network.get_action(obs, actions)
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        q_values = (old_pmfs * self.q_network.atoms).sum(1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
```
