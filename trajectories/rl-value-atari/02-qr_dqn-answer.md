**Problem (from step 1).** Double DQN removed the non-uniform overestimation but its critic is still a
single scalar mean per action — a point estimate with no notion of its own spread. On Seaquest (18
actions, long horizon) the residual bias and the discarded return-spread show up as wide seed-to-seed
variance (5386–7804). The ceiling is representational: collapsing the return to its mean throws away
exactly the information that varies seed to seed.

**Key idea — transpose the parametrization.** The return $Z(x,a)$ obeys a distributional Bellman
recursion contracting in Wasserstein, but Wasserstein cannot be minimized from samples (biased
gradients). The fixed-atom route (learn probabilities on a prescribed grid, KL after projection) needs
$[V_{\min},V_{\max}]$ and a projection. Instead fix the probabilities to uniform $1/N$ and learn the
*locations* $Z_\theta(x,a)=\frac1N\sum_i\delta_{\theta_i(x,a)}$ — the locations are *quantiles* of the
return. No support bounds, no projection, support that adapts per state (Seaquest's thousands vs Pong's
tens).

**Which quantiles, and the loss.** Minimizing $W_1$ cell-by-cell puts the locations at the midpoint
quantiles $\hat\tau_i=\frac{2i-1}{2N}$. Hit them with quantile regression $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$
whose sign-only gradient is unbiased from one sample; Huberize ($\kappa=1$) to kill the kink at $u=0$,
weighted by $|\tau-\mathbb{1}_{u<0}|$. Train all pairs of predicted locations $\theta_i(x,a)$ against
bootstrapped target locations $\mathcal{T}\theta_j=r+\gamma\theta_j(x',a^\star)$, summed over $i$,
averaged over $j$.

**Why it works.** The critic now models the spread instead of letting it corrupt one scalar; the
Huber's tail-clipping and the distributional averaging stabilize the high-return Seaquest transitions
that yanked the squared-loss mean around. Greedy control is unchanged — act on the per-action mean
$\frac1N\sum_i\theta_i$ — so this isolates the effect of the richer critic.

**Scaffold edit / hyperparameters.** Fixed `NatureDQNEncoder` → 512 → `Linear(512, n_actions * N)`,
reshaped to $(B, n_{\text{actions}}, N)$; `forward` returns the per-action mean so eval-argmax is
unchanged. $N=200$ (the budget-check reference head), $\kappa=1$ hard-coded (no $\kappa=0$ branch).
Adam at `args.learning_rate = 1e-4` with $\epsilon_{\text{Adam}}=0.01/\text{batch\_size}$; hard target
copy every `target_network_frequency` steps.

**What to watch.** Pong stays $\approx21$; Breakout should clear 170 (mid-200s) and Seaquest clear 6789
(into the 9000s), with seed-to-seed variance *not* blowing up as the mean rises. If the mean climbs but
the spread widens, $N=200$ under-resolves the tails — and the next move is to stop fixing the quantile
levels at all.

```python
class QNetwork(nn.Module):
    """QR-DQN quantile Q-network: NatureDQNEncoder (fixed) + quantile head."""

    def __init__(self, envs, n_quantiles=200):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n = envs.single_action_space.n
        self.encoder = NatureDQNEncoder()
        self.head = nn.Linear(ENCODER_FEATURE_DIM, self.n * n_quantiles)

    def forward(self, x):
        """Return Q-values as mean of quantile values per action."""
        features = self.encoder(x)
        quantiles = self.head(features).view(len(x), self.n, self.n_quantiles)
        q_values = quantiles.mean(dim=2)
        return q_values

    def get_quantiles(self, x):
        """Return raw quantile values: [batch, n_actions, n_quantiles]."""
        features = self.encoder(x)
        return self.head(features).view(len(x), self.n, self.n_quantiles)


class ValueAlgorithm:
    """QR-DQN -- Quantile Regression DQN with distributional value learning."""

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        self.target_network_frequency = args.target_network_frequency
        self.n_quantiles = 200
        self.kappa = 1.0  # Huber loss threshold

        self.q_network = QNetwork(envs, n_quantiles=self.n_quantiles).to(device)
        self.target_network = QNetwork(envs, n_quantiles=self.n_quantiles).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N) for i = 1, ..., N
        self.tau = torch.arange(1, self.n_quantiles + 1, dtype=torch.float32, device=device)
        self.tau = (2 * self.tau - 1) / (2 * self.n_quantiles)

    def select_action(self, obs, epsilon):
        """Greedy action selection using mean of quantile values."""
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def update(self, batch, global_step):
        """QR-DQN update: quantile Huber loss."""
        with torch.no_grad():
            # Get quantile values for next state from target network
            next_quantiles = self.target_network.get_quantiles(batch.next_observations)  # [batch, n_actions, N]
            next_q = next_quantiles.mean(dim=2)  # [batch, n_actions]
            next_actions = next_q.argmax(dim=1)  # [batch]
            # Select quantiles for best actions
            next_quantiles_best = next_quantiles[torch.arange(len(batch.next_observations)), next_actions]  # [batch, N]
            # Compute target quantile values
            target_quantiles = batch.rewards + self.gamma * next_quantiles_best * (1 - batch.dones)

        # Get current quantile values for taken actions
        current_quantiles_all = self.q_network.get_quantiles(batch.observations)  # [batch, n_actions, N]
        current_quantiles = current_quantiles_all[torch.arange(len(batch.observations)), batch.actions.flatten()]  # [batch, N]

        # Quantile Huber loss
        # Pairwise TD errors: [batch, N (pred), N (target)]
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

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

        # Hard target update
        if global_step % self.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        q_values = current_quantiles.mean(dim=1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
```
