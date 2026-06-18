**Problem.** Dueling DQN improved the *mean* state-value estimate but still learns only a scalar
`Q = E[Z]`. On LunarLander the return is bimodal (safe landing vs crash), so the mean corresponds to no
real outcome and the greedy argmax cannot tell a reliably-landing action from a high-variance one — the
seed that fell into the crash basin (−89.25) is that failure. The fix is to stop collapsing the return
to its mean.

**Key idea.** Learn the **return distribution** via quantile regression. The distributional Bellman
operator contracts in Wasserstein (not KL/TV, which are blind to the `γ`-shrink of disjoint supports),
but sampled Wasserstein has a biased gradient. Escape: parametrize the distribution as `N` equal-mass
Diracs at learnable **locations** `θ_i` — the locations are quantiles. The `W_1`-optimal locations are
the **midpoint quantiles** `τ̂_i = (2i−1)/(2N)`, and the **quantile (Huber) loss**
`ρ_τ^κ(u) = |τ − 1{u<0}| · L_κ(u)` hits them with an *unbiased* sign-only gradient. No fixed support,
no projection.

**Why it should help.** Modeling the distribution lets the greedy policy (still mean-greedy,
`argmax_a mean_i θ_i`) distinguish high-mean-because-reliable from high-mean-because-volatile —
expected to lift LunarLander's worst seed out of the negative basin. Little to gain on near-unimodal
Acrobot or already-saturated CartPole.

**Scaffold-specific choices.** Head = linear `84 → |A|·N` on the **fixed MLP encoder**;
`forward` returns the per-action quantile mean so the harness argmaxes a clean `(batch, |A|)`. **`N=50`**
(not the larger generic value — coarse-but-stable for classic control on a single 500k-step run).
Target = scaffold **plain-DQN style** (select `a*` and evaluate quantiles both on the **target** net,
not double-DQN). `κ=1`. All-pairs quantile Huber loss. No dueling head — isolate the distributional
effect.

**Hyperparameters.** `n_quantiles=50`; `kappa=1.0`; `tau_i=(2i−1)/(2N)`; Adam `lr=2.5e-4`;
`gamma=0.99`; everything else the frozen DQN loop.

```python
# EDITABLE region of custom_value_discrete.py — step 2: QR-DQN
class QNetwork(nn.Module):
    """Quantile Q-network for QR-DQN: MLPEncoder (fixed) + n_actions x n_quantiles head."""

    def __init__(self, obs_dim, n_actions, n_quantiles=50):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.encoder = MLPEncoder(obs_dim)
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions * n_quantiles)

    def forward(self, obs):
        """Return Q-values as mean of quantile values per action."""
        features = self.encoder(obs)
        quantiles = self.head(features).view(len(obs), self.n_actions, self.n_quantiles)
        q_values = quantiles.mean(dim=2)
        return q_values

    def get_quantiles(self, obs):
        """Return raw quantile values: [batch, n_actions, n_quantiles]."""
        features = self.encoder(obs)
        return self.head(features).view(len(obs), self.n_actions, self.n_quantiles)


class ValueAlgorithm:
    """QR-DQN -- Quantile Regression DQN with distributional value learning."""

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.n_quantiles = 50
        self.kappa = 1.0  # Huber loss threshold
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions, self.n_quantiles).to(device)
        self.target_network = QNetwork(obs_dim, n_actions, self.n_quantiles).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N) for i = 1, ..., N
        self.tau = torch.arange(1, self.n_quantiles + 1, dtype=torch.float32, device=device)
        self.tau = (2 * self.tau - 1) / (2 * self.n_quantiles)

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
            # Get quantile values for next state from target network
            next_quantiles = self.target_network.get_quantiles(next_obs)  # [batch, n_actions, n_quantiles]
            next_q = next_quantiles.mean(dim=2)  # [batch, n_actions]
            next_actions = next_q.argmax(dim=1)  # [batch]
            # Select quantiles for best actions
            next_quantiles_best = next_quantiles[torch.arange(len(next_obs)), next_actions]  # [batch, n_quantiles]
            # Compute target quantile values
            target_quantiles = rewards.unsqueeze(1) + self.gamma * next_quantiles_best * (1 - dones.unsqueeze(1))

        # Get current quantile values for taken actions
        current_quantiles = self.q_network.get_quantiles(obs)  # [batch, n_actions, n_quantiles]
        current_quantiles = current_quantiles[torch.arange(len(obs)), actions]  # [batch, n_quantiles]

        # Quantile Huber loss
        # Pairwise TD errors: [batch, n_quantiles (pred), n_quantiles (target)]
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)  # [batch, N, N]

        # Huber loss element-wise
        abs_td = td_errors.abs()
        huber = torch.where(abs_td <= self.kappa,
                            0.5 * td_errors ** 2,
                            self.kappa * (abs_td - 0.5 * self.kappa))

        # Asymmetric weighting by quantile level
        tau = self.tau.view(1, -1, 1)
        quantile_weights = torch.abs(tau - (td_errors < 0).float())
        loss = (quantile_weights * huber / self.kappa).sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        q_values = current_quantiles.mean(dim=1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
```
