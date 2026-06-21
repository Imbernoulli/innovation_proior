The quantile critic did what I bet it would. QR-DQN cleared the decoupled-mean rung on both informative games — Breakout $252$ (seeds $246.9/185.7/324.7$) against Double DQN's $170.7$, Seaquest $9027$ ($7892/7038/12152$) against $6789$ — and Pong held at the ceiling ($20.9$). So modeling the spread instead of collapsing it paid off exactly where I argued. But look harder at the Seaquest seeds: $7038$, $7892$, and then a $12152$ that is nearly double the lowest. The mean rose, but the spread did not collapse — one seed found a much heavier upper tail than the others. That is not the bias variance I was chasing before; it is the critic telling me about its *resolution*. I picked the number $200$ by hand, and the network is stuck with it forever: the output layer is $|\mathcal A|\times200$, those $200$ numbers are tied one-to-one to $200$ levels $\hat\tau_i=(2i-1)/(2N)$ fixed before training, every update touches the same comb, and the approximation can never be finer than $200$ no matter what the frozen encoder has left to give. On Seaquest, whose return distribution runs into five figures with a long heavy upper tail, a $200$-step staircase resolves exactly the tail the $12152$ seed lived in most coarsely. The resolution is an architectural constant, not something that improves with capacity or training. That is the one structural limit left on the value object, and removing it is the natural next move.

What am I really approximating? The $200$ locations are quantiles, and "the location of the $\tau$-quantile as a function of $\tau$" is the quantile function $F_Z^{-1}(\tau)$ — the inverse CDF, a map from a probability $\tau\in[0,1]$ to a return value. The quantile agent samples this function at $200$ fixed grid points and learns those $200$ outputs. But $F_Z^{-1}$ is a *continuous* function of $\tau$. I propose **IQN** — Implicit Quantile Networks — which learns the whole function: a network $Z_\tau(x,a):=F_{Z(x,a)}^{-1}(\tau)$ that takes a base sample $\tau\sim U([0,1])$ and returns a genuine sample of the return. That last sentence is the reparameterization trick stated for returns — push a uniform through the quantile function and you get a return sample — so instead of learning probabilities on fixed locations, or locations on fixed probabilities, I learn a deterministic function that *reparameterizes* a base sample into a return. The distribution is defined implicitly: I never write down its density or CDF, I only know how to sample it (draw $\tau$, evaluate the net). With enough capacity this approximates any return distribution, because any distribution is the pushforward of a uniform through its own quantile function, and the fidelity is now set by how well the network fits a curve — a matter of capacity and training, not a hardcoded $N$. That is what frees Seaquest's tail: where the $200$-comb spent the same resolution everywhere, an implicit function can bend sharply in the upper quantiles where the data demands it, and the quantile count becomes a *sample count per update*, chosen freely.

The training is the part I am surest of, because it is the same quantile-regression machinery the last rung used, and that machinery never cared that the levels came from a grid — only that I am trying to hit the $\tau$-quantile. For a single $\tau$, the $\tau$-quantile minimizes $\mathbb{E}_{\hat Z}[\rho_\tau(\hat Z-\theta)]$ with $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$, whose error-gradient $\tau-\mathbb{1}_{u<0}$ depends only on the sign of $u$, so one sample is unbiased. The Bellman version uses *two independent* base samples $\tau$ and $\tau'$:
$$\delta^{\tau,\tau'}=r+\gamma\,Z_{\tau'}(x',\pi(x'))-Z_\tau(x,a),$$
where $Z_\tau(x,a)$ is my prediction at level $\tau$ and $Z_{\tau'}(x',\pi(x'))$ is a bootstrapped target-network sample at level $\tau'$, regressing the prediction at level $\tau$ toward the target at level $\tau'$ with the quantile loss at level $\tau$. The decisive difference from QR-DQN is that $\tau,\tau'$ are drawn *fresh* from a continuous distribution every update, not read off a fixed comb — so the function is supervised at a dense, ever-changing set of levels, which is what forces it to be right *as a function* and not just at $200$ pinned points. The kink in $\rho_\tau$ at $u=0$ hurts the net the same way it did before, so I Huberize with threshold $\kappa$ — quadratic inside $[-\kappa,\kappa]$, linear outside — weighted by $|\tau-\mathbb{1}\{u<0\}|$, with $\kappa=1$ recovering the gradient-clipped squared error inside the band but asymmetric per level. The sample counts are now knobs: $N$ prediction samples $\tau_i$ and $N'$ target samples $\tau'_j$ give the all-pairs loss $\mathcal L=\frac1{N'}\sum_{i=1}^{N}\sum_{j=1}^{N'}\rho^\kappa_{\tau_i}(\delta^{\tau_i,\tau'_j})$ — sum over predicted levels $i$ (each regressed at its own level), average over target samples $j$ (a Monte-Carlo estimate of the bootstrapped distribution, hence $1/N'$). $N$ is how much of my own quantile function I touch per update; $N'$ is a variance-reduction count on the target. At $N=1$ I touch a single random point of the curve per update — a clean diagnostic that the benefit is the implicit representation, not merely many heads. For the working agent I want both moderate, $N=N'=8$, costing $8\times8=64$ pairwise terms, comparable to the grid agent's work and far below its $200$-wide output.

The architecture is where the design choices bite, because the encoder is frozen and I have to inject $\tau$ through the only thing I control. The state side stays as close as possible to the fixed machinery: $\psi(x)$ is the frozen `NatureDQNEncoder` ending in its $512$-dim ReLU feature vector, and the action head is a single linear map from $512$ features to actions. I add a function $\phi(\tau)$ embedding the scalar level into the same $512$-dim space and combine it with $\psi(x)$ before the head. The combination matters. Concatenation is the lazy choice, but with a single linear head it makes $\tau$ enter only *additively* — the output is (linear in $\psi$) + (linear in $\phi(\tau)$), so a $\tau$-dependent term can only slide the whole curve up and down, never reshape the quantile function per state, which is the entire point. So I make $\tau$ *multiply* the state features via the Hadamard product,
$$Z_\tau(x,a)\approx f\big(\psi(x)\odot\phi(\tau)\big)_a,$$
so $\phi(\tau)$ gates each feature of $\psi(x)$ — feature-wise multiplicative modulation that lets even a single linear head see a genuinely $\tau$-conditioned input. This matters more here than in a from-scratch IQN precisely because the encoder is frozen: I cannot adapt $\psi$ to make room for $\tau$, so the interaction has to be forced through the shallow head, and multiplication forces it. For the embedding itself, feeding the raw scalar through a linear layer is rank-one in $\tau$ and cannot represent rich dependence, so I use a basis expansion — cosines of increasing frequency, the natural bounded Fourier-type basis on an interval — lifting $\tau$ into $n$ features $\cos(\pi i\tau)$, $i=1,\dots,n$, then a linear-then-ReLU into the $512$-space, $\phi_j(\tau)=\mathrm{ReLU}(\sum_{i=1}^n\cos(\pi i\tau)\,w_{ij}+b_j)$, with $n=64$ cosines: plenty of frequency content to describe how the quantile function bends, and cheap. The accounting is decisively in my favor for this task. The encoder is untouched; the only new parameters are the cosine embedding's `Linear(64, 512)` and a head that now maps modulated features to $|\mathcal A|$ actions rather than $|\mathcal A|\times200$. I have *removed* the $200$-fold output blowup — the very head the scaffold's budget check is sized around — and replaced it with one small embedding branch, leaving the IQN head plus embedding a few tens of thousands of parameters, well inside the $1.05\times$-QR-DQN budget on all three action-space sizes ($4$, $6$, $18$). The representational gain comes from learning a function of $\tau$, not from width.

For control I keep the same risk-neutral objective QR-DQN maximized, so any difference is attributable to the implicit representation alone. The mean is $\mathbb{E}_{\tau\sim U([0,1])}[Z_\tau(x,a)]$, which I approximate with $K$ fresh samples and argmax: $\tilde\pi(x)=\arg\max_a\frac1K\sum_k Z_{\tilde\tau_k}(x,a)$, $\tilde\tau_k\sim U([0,1])$, $K=32$ — enough to make the choice a Monte-Carlo estimate of the mean rather than one noisy draw. I deliberately stay risk-neutral even though the implicit quantile function unlocks more: sampling $\tau$ from a non-uniform $\mu$ would yield a distorted expectation $\int_0^1 F^{-1}_Z(u)\,d\mu(u)$ — mass near low quantiles giving risk-aversion (CVaR$(\eta)$ for $u\sim U([0,\eta])$), mass near high quantiles risk-seeking — but using it would change *what* is being maximized, and I want a clean comparison. Fitting the edit surface, `q_network.forward(x)` returns the $K$-sample mean over $\tau$ so the loop's eval-argmax is unchanged, while `get_quantiles(x, taus)` returns the raw $(B,|\mathcal A|,M)$ samples for the loss; `update` samples $N=8$ prediction $\tau$ and $N'=8$ target $\tau'$, picks the next action greedily on the target net's $K$-sample mean, forms the pairwise TD errors and the all-pairs quantile Huber, steps Adam at the scaffold's `learning_rate = 1e-4` with $\epsilon_{\text{Adam}}=0.01/\text{batch\_size}$, and does the hard target copy at `target_network_frequency`. Against QR-DQN's real numbers — Breakout $252.4$ (max seed $324.7$), Seaquest $9027$ (max seed $12152$), Pong $20.9$ — I expect IQN to hold Pong at $\approx21$ (a drop would mean the cosine modulation destabilized the easy case), to match or beat the means on Breakout and Seaquest while removing the fixed-resolution bottleneck, and — the falsifiable claim — to gain *most* on Seaquest, the game whose return distribution is widest and whose long upper tail the $200$-comb resolved most coarsely. If Breakout improves but Seaquest does not, the bottleneck was never resolution and I misread the $12152$ seed; if IQN merely matched QR-DQN everywhere, $200$ quantiles already saturated these games and the implicit representation buys nothing at this budget.

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
