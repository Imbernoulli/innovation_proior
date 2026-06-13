**Problem (from step 2).** QR-DQN broke the scalar ceiling — Breakout 252, Seaquest 9027, Pong 21,
clearing Double DQN on the two informative games — but its resolution is an *architectural constant*.
The network emits exactly $N=200$ values per action, tied one-to-one to the $200$ quantile *levels*
$\hat\tau_i=(2i-1)/(2N)$ chosen before training. The approximation can never be finer than $N$ without
widening the output layer and retraining, every update touches the same fixed comb of levels, and the
policy still reads only the mean.

**Key idea — learn the quantile function implicitly.** The $200$ locations are samples of the quantile
function $F_Z^{-1}(\tau)$ at fixed $\tau$. Learn the *whole function*: a network $Z_\tau(x,a)$ that
takes a base sample $\tau\sim U([0,1])$ and returns a return sample (the reparameterization trick for
returns). The distribution is defined implicitly — sample $\tau$, evaluate the net — so its fidelity is
bounded by capacity and training, not by a hardcoded $N$. The number of quantiles becomes a *sample
count per update*, chosen freely.

**Architecture.** Keep the fixed `NatureDQNEncoder` $\psi(x)$ → 512, and a single linear action head
$f$. Embed $\tau$ into the same 512-space with a cosine basis and inject it *multiplicatively* so a
shallow head still sees a state-dependent $\tau$ interaction:
$Z_\tau(x,a)\approx f(\psi(x)\odot\phi(\tau))_a$, $\phi_j(\tau)=\mathrm{ReLU}(\sum_{i=1}^{n}\cos(\pi i\tau)w_{ij}+b_j)$,
$n=64$. Concatenation would let $\tau$ enter only as an additive shift of the whole curve; the Hadamard
product forces feature-wise modulation. The only new parameters are the cosine embedding
`Linear(64,512)` and the head shrinks from $|\mathcal A|\times200$ to $|\mathcal A|$ — well under the
QR-DQN budget.

**Loss.** Two independent base samples per term:
$\delta^{\tau,\tau'}=r+\gamma Z_{\tau'}(x',\pi(x'))-Z_\tau(x,a)$. All-pairs quantile Huber over $N$
prediction samples $\tau_i$ and $N'$ target samples $\tau'_j$, $\kappa=1$, summed over $i$, averaged
over $j$. Levels are sampled *fresh* each update (continuous, not a grid), so the function is supervised
across changing probability levels. $N=N'=8$.

**Policy.** Mean-greedy by Monte Carlo: $\arg\max_a\frac1K\sum_k Z_{\tilde\tau_k}(x,a)$, $K=32$,
$\tilde\tau_k\sim U([0,1])$ — risk-neutral, the same objective QR-DQN maximized, so the comparison
isolates the implicit continuous representation. (A non-uniform $\tau$-sampling would yield distorted
expectations / risk-sensitive policies; not used here.)

**Scaffold edit / hyperparameters.** `forward(x)` returns the $K$-sample mean Q-values so the loop's
eval-argmax is unchanged; `get_quantiles(x, taus)` returns $(B, n_{\text{actions}}, M)$. $n=64$,
$N=N'=8$, $K=32$, $\kappa=1$. Adam at `args.learning_rate = 1e-4`,
$\epsilon_{\text{Adam}}=0.01/\text{batch\_size}$; hard target copy every `target_network_frequency`
steps.

**What to clear.** QR-DQN's strongest-baseline numbers: Breakout mean 252.4 (max seed 324.7), Seaquest
mean 9027 (max seed 12152), Pong 20.9. IQN should match or beat the means on Breakout and Seaquest while
removing the fixed-resolution bottleneck, and hold Pong at its ceiling. I would validate that (1) the
$K$-sample mean policy is stable across seeds (no resolution-induced variance), (2) Seaquest — where the
return distribution is widest and the 200-comb left the most unmodeled tail — is where the implicit
representation gains most, and (3) the head stays within the parameter budget on all three action-space
sizes (4 / 6 / 18).

```python
class QNetwork(nn.Module):
    """IQN quantile network: NatureDQNEncoder (fixed) + cosine tau-embedding + linear head.

    Z_tau(x, a) = head( psi(x) (Hadamard) phi(tau) )_a, with psi the fixed 512-dim
    encoder features and phi a ReLU cosine embedding of the base sample tau.
    """

    N_COS = 64       # cosine embedding dimension
    K_POLICY = 32    # samples for the mean-greedy policy / Q-value readout

    def __init__(self, envs):
        super().__init__()
        self.n = envs.single_action_space.n
        self.encoder = NatureDQNEncoder()
        self.phi = nn.Linear(self.N_COS, ENCODER_FEATURE_DIM)   # cosine basis -> feature dim
        self.head = nn.Linear(ENCODER_FEATURE_DIM, self.n)      # single linear head to actions
        # frequencies i = 1..N_COS used in cos(pi * i * tau)
        self.register_buffer(
            "freqs", torch.arange(1, self.N_COS + 1, dtype=torch.float32) * 3.141592653589793
        )

    def get_quantiles(self, x, taus):
        """Z_tau(x, .) for each of M sampled taus -> [batch, n_actions, M]."""
        psi = self.encoder(x).unsqueeze(1)                      # [B, 1, d]
        cos = torch.cos(taus.unsqueeze(-1) * self.freqs)        # [B, M, N_COS]
        phi = F.relu(self.phi(cos))                             # [B, M, d]
        feats = psi * phi                                       # Hadamard modulation [B, M, d]
        return self.head(feats).permute(0, 2, 1)               # [B, n_actions, M]

    def forward(self, x):
        """Q-values as a K-sample Monte-Carlo estimate of the per-action mean E_tau[Z_tau]."""
        taus = torch.rand(len(x), self.K_POLICY, device=x.device)
        return self.get_quantiles(x, taus).mean(dim=2)         # [B, n_actions]


class ValueAlgorithm:
    """IQN -- Implicit Quantile Network with an implicitly-defined return distribution."""

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        self.target_network_frequency = args.target_network_frequency
        self.n_quantiles = 8        # N : prediction samples per update
        self.n_quantiles_tgt = 8    # N': target samples per update
        self.kappa = 1.0            # Huber threshold

        self.q_network = QNetwork(envs).to(device)
        self.target_network = QNetwork(envs).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)

    def select_action(self, obs, epsilon):
        """Greedy action selection on the K-sample Monte-Carlo mean."""
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def update(self, batch, global_step):
        """IQN update: all-pairs quantile Huber loss on freshly sampled tau levels."""
        b = batch.observations.shape[0]
        idx = torch.arange(b, device=self.device)

        with torch.no_grad():
            # mean-greedy next action on the target network (K-sample MC mean)
            next_q = self.target_network(batch.next_observations)               # [B, n_actions]
            next_actions = next_q.argmax(dim=1)                                 # [B]
            taus_tgt = torch.rand(b, self.n_quantiles_tgt, device=self.device)  # tau' ~ U([0,1])
            z_next = self.target_network.get_quantiles(batch.next_observations, taus_tgt)  # [B, A, N']
            z_next = z_next[idx, next_actions]                                  # [B, N']
            not_done = (1 - batch.dones.flatten()).view(b, 1)
            target_z = batch.rewards.flatten().view(b, 1) + self.gamma * not_done * z_next  # [B, N']

        taus = torch.rand(b, self.n_quantiles, device=self.device)             # tau ~ U([0,1])
        z = self.q_network.get_quantiles(batch.observations, taus)            # [B, A, N]
        theta = z[idx, batch.actions.flatten()]                                # [B, N] predicted

        # pairwise TD errors u_ij = T theta_j - theta_i : [B, N (pred), N' (target)]
        td_errors = target_z.unsqueeze(1) - theta.unsqueeze(2)

        abs_td = td_errors.abs()
        huber = torch.where(abs_td <= self.kappa,
                            0.5 * td_errors ** 2,
                            self.kappa * (abs_td - 0.5 * self.kappa))
        # asymmetric weight by the prediction quantile level tau_i (along the i-axis)
        tau = taus.unsqueeze(2)                                                # [B, N, 1]
        quantile_weights = torch.abs(tau - (td_errors.detach() < 0).float())
        # average over target samples j (1/N'), sum over prediction quantiles i, mean over batch
        loss = (quantile_weights * huber / self.kappa).mean(dim=2).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard target update
        if global_step % self.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"td_loss": loss.item(), "q_values": theta.mean().item()}
```
