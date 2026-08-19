Value-based deep RL on Atari learns one number per state-action, the expected return $Q(x,a)=\mathbb{E}[Z(x,a)]$, and acts greedily on it. Distributional RL shows that learning the whole law of the random return $Z(x,a)$ helps even when the policy still reads only its mean, because the richer target shapes the representation that feeds the value estimate. But every distributional agent I have to work with makes a discrete commitment before training begins. The categorical agent (C51) pins the *return locations* to a fixed comb $z_1<\cdots<z_N$ on a prescribed interval $[V_{\min},V_{\max}]$, learns the probabilities on those atoms, projects the shifted Bellman target back onto the comb, and minimizes a KL after projection — so it has to be told the return range in advance, its resolution is capped at $N$ atoms placed a priori, and it optimizes a Cramér-geometry surrogate rather than the Wasserstein metric the distributional Bellman operator actually contracts in. The quantile agent (QR-DQN) transposes this: it fixes the probabilities to a uniform $1/N$ and learns the *locations*, which become quantiles of the return at the predetermined midpoint levels $\hat\tau_i=(2i-1)/(2N)$, trained by quantile regression whose sample gradient is unbiased and whose projected operator contracts in $\infty$-Wasserstein. That removes the support bounds and the projection, but it does not remove the wart that bothers me: the network emits exactly $N$ values per action, tied one-to-one to $N$ levels chosen before training started. A $200$-Dirac approximation is a staircase with $200$ steps no matter how much spare capacity the rest of the net has; to make it finer I must widen the output layer and retrain from scratch. The resolution of the distribution is an *architectural constant*, not something that improves with capacity or training. And in both families the policy still reads only the mean, discarding everything the distribution knows about the spread of outcomes.

So I ask what object the quantile agent is really approximating. Its locations are quantiles, and "the location of the $\tau$-quantile as a function of $\tau$" is just the quantile function $F_Z^{-1}(\tau)$, the inverse CDF mapping a probability $\tau\in[0,1]$ to a return value. The grid agent samples this continuous function at $N$ fixed points and learns those $N$ outputs. I propose instead to learn the whole curve — a map I can query at *any* $\tau$ — so that fidelity is set by how well a network approximates a function, a matter of capacity and training, not a hardcoded $N$. The method is IQN, the implicit quantile network. The target I want is $Z_\tau(x,a):=F^{-1}_{Z(x,a)}(\tau)$, so that for $\tau\sim U([0,1])$ the value $Z_\tau(x,a)$ is a genuine sample from the return distribution. That single sentence is the reparameterization trick stated for returns: push a uniform random variable through the quantile function and you get a sample of the return. The distribution is therefore defined *implicitly* — I never write its density or CDF, I only know how to sample it: draw $\tau$, evaluate the network. With enough capacity this approximates any return distribution, because any distribution is the pushforward of a uniform through its own quantile function. The number of quantiles stops being an output-layer width and becomes a *sample count per update*, chosen freely.

Two design questions follow that the grid agent answered by construction and I now must answer by design: what trains the function so $Z_\tau(x,a)$ lands on $F^{-1}_{Z(x,a)}(\tau)$ for *every* $\tau$, and how does $\tau$ enter the network. The training I am sure about, because the quantile-regression argument never cared that the grid agent sampled $\tau$ from a comb — it cares only that I am trying to hit the $\tau$-quantile. For any single level $\tau$, the $\tau$-quantile minimizes the asymmetric loss $\mathbb{E}_{\hat Z}[\rho_\tau(\hat Z-\theta)]$ with $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$, whose error-gradient $\tau-\mathbb{1}_{u<0}$ depends only on the sign of the residual, so one sample gives an unbiased gradient. The Bellman version builds the sampled TD error from *two independent* base samples $\tau$ and $\tau'$,
$$\delta^{\tau,\tau'}_t=r_t+\gamma\,Z_{\tau'}\big(x_{t+1},\pi(x_{t+1})\big)-Z_{\tau}(x_t,a_t),$$
regressing the network's prediction at level $\tau$ toward a bootstrapped target-network sample at level $\tau'$ using the quantile loss at level $\tau$. The point is that $\tau$ and $\tau'$ are drawn fresh from a continuous distribution each update, never from a fixed comb, so the function is supervised on a dense, ever-changing set of levels — which is exactly what forces it to be right *as a function* rather than at $200$ pinned points. The kink of $\rho_\tau$ at $u=0$ hurts a deep net, because the gradient magnitude stays at $\tau$ or $1-\tau$ right down to zero error and the locations jitter, so I Huberize as the grid agent did: quadratic inside $[-\kappa,\kappa]$, linear outside,
$$\mathcal L_\kappa(u)=\begin{cases}\tfrac12u^2,&|u|\le\kappa,\\[2pt]\kappa(|u|-\tfrac12\kappa),&|u|>\kappa,\end{cases}\qquad
\rho^\kappa_\tau(u)=\big|\tau-\mathbb{1}\{u<0\}\big|\,\frac{\mathcal L_\kappa(u)}{\kappa},$$
with $\kappa=1$. The division by $\kappa$ keeps the large-error slope independent of the transition width, so changing $\kappa$ moves where the quadratic hands off to the linear part rather than rescaling the whole loss. Because the quantile counts are now knobs, I sample $N$ levels $\tau_i$ for the predictions and $N'$ levels $\tau'_j$ for the targets and sum over all pairs,
$$\mathcal L=\frac1{N'}\sum_{i=1}^{N}\sum_{j=1}^{N'}\rho^\kappa_{\tau_i}\big(\delta^{\tau_i,\tau'_j}_t\big).$$
I sum over the predicted levels $i$ because each predicted quantile is regressed at its own level, and average over the target samples $j$ (hence the $1/N'$) because they are a Monte-Carlo estimate of the bootstrapped distribution. $N$ controls how much of my own quantile function I shape per step; $N'$ is a variance-reduction count denoising the regression target, so its marginal value drops once the target estimate is quiet. Pushing $N=1$ touches a single random point per update — a clean diagnostic for whether the gain is merely an auxiliary-loss effect of having many heads — but for the working agent I want both moderate and nowhere near a fixed $200$, landing on $N=N'=8$, whose $8\times8=64$ pairwise terms cost about what the grid agent costs per update while far below its $200$-wide output.

The architecture question is how $\tau$ enters without rebuilding the DQN machinery. I take $\psi(x)$ to be the fixed Nature-DQN encoder ending in the $512$-dimensional ReLU feature vector — choosing the boundary so the encoder includes the usual hidden layer and the action head $f$ is a single linear map from $512$ features to actions. I add a third function $\phi(\tau)$ that embeds the scalar level into the same $512$-dimensional space and combine it with $\psi(x)$ before the head. The combination matters. If I merely concatenate $\psi(x)$ and $\phi(\tau)$ and hit them with one linear map, $\tau$ enters only *additively*: the output is (linear in $\psi$) plus (linear in $\phi(\tau)$), so $\tau$ can only slide the whole curve up and down — a state-independent shift — and cannot reshape the quantile function per state. The shape of $F^{-1}$ has to change with $\tau$ in a state-dependent way, so I make $\tau$ *multiply* the state features through the element-wise (Hadamard) product,
$$Z_\tau(x,a)\approx f\big(\psi(x)\odot\phi(\tau)\big)_a,$$
letting $\phi(\tau)$ gate each feature of $\psi(x)$ so even a single linear $f$ on top sees a genuinely $\tau$-conditioned input. For the embedding itself, feeding the raw scalar $\tau$ through a linear layer is rank-one in $\tau$ and too weak; I want a basis expansion that lifts the scalar into many features varying at different rates, so a linear layer on top can synthesize an arbitrary smooth function of $\tau$. Cosines of increasing frequency are the natural bounded basis on an interval, so I expand $\tau$ into $n$ cosine features $\cos(\pi i\tau)$ and pass them through a linear-then-ReLU into the feature dimension,
$$\phi_j(\tau)=\operatorname{ReLU}\!\Big(\sum_{i=1}^{n}\cos(\pi i\tau)\,w_{ij}+b_j\Big),\qquad n=64,$$
where indexing the basis from $0$ instead only adds a constant cosine that the bias $b_j$ absorbs. The linear-then-ReLU lets the head pick and recombine frequencies and gives $\phi$ the same nonlinearity budget as the rest of the head, at the cost of one tiny cosine expansion plus one linear layer shared across all $\tau$ samples in a batch. This adds no capacity that defeats the point: the torso is untouched, the head is the same shallow $f$, and where the grid agent's output was $|\mathcal A|\times N$, mine is $|\mathcal A|$ per evaluated $\tau$ reused across samples — I have *removed* the $N$-fold output blowup and replaced it with a small embedding branch.

The policy acts on the mean by Monte Carlo, since $\mathbb{E}[Z(x,a)]=\mathbb{E}_{\tau\sim U([0,1])}[Z_\tau(x,a)]$, approximated with $K$ fresh samples,
$$\tilde\pi(x)=\arg\max_a\frac1K\sum_{k=1}^K Z_{\tilde\tau_k}(x,a),\qquad \tilde\tau_k\sim U([0,1]),\ K=32,$$
and the bootstrapped target uses this same greedy action with $\gamma=0$ at terminals; the replay buffer, $\epsilon$-greedy, and periodic target copy stay as in DQN. Having the whole quantile function unlocks something the mean throws away. The mean is $\int_0^1 F_Z^{-1}(u)\,du$, a *uniform* average over levels, and nothing forces the policy to weight levels uniformly. If I instead sample levels from a distribution $\mu$ on $[0,1]$ and act on $\frac1K\sum_k Z_{u_k}(x,a)$ with $u_k\sim\mu$, the limit estimates the distorted expectation
$$Q_\mu(x,a)=\int_0^1 F^{-1}_{Z(x,a)}(u)\,d\mu(u),$$
an integral of the quantile function against a non-uniform weighting over probability levels. Mass near low quantiles makes the policy risk-averse, mass near high quantiles risk-seeking; conditional value-at-risk at level $\eta$ is the clean lower-tail case $u\sim U([0,\eta])$, estimating $\frac1\eta\int_0^\eta F_Z^{-1}(u)\,du$ and so caring only about the worst $\eta$-fraction of outcomes. The risk-neutral agent is just $\mu=U([0,1])$, and I keep it for the value-maximizing comparison so the only change against the grid agent is the implicit continuous representation, with the risk knob a bonus the representation grants by changing nothing but how $\tau$ is sampled.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

N_QUANTILES = 8       # N: prediction samples per update
N_QUANTILES_TGT = 8   # N': target samples per update
K_POLICY = 32         # samples for the mean-greedy policy
N_COS = 64            # cosine embedding dimension n
KAPPA = 1.0


class ConvTorso(nn.Module):
    """Nature-DQN conv stack: (B,4,84,84) -> (B, d)."""
    def __init__(self, d=512):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, d), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


class ImplicitQuantileNetwork(nn.Module):
    """Z_tau(x,a) = f( psi(x) (Hadamard) phi(tau) )_a, with phi a cosine embedding of tau."""
    def __init__(self, n_actions, d=512, n_cos=N_COS):
        super().__init__()
        self.n_actions, self.d, self.n_cos = n_actions, d, n_cos
        self.torso = ConvTorso(d)
        self.phi = nn.Linear(n_cos, d)        # cosine basis -> feature dim
        self.head = nn.Linear(d, n_actions)   # f: single linear head to actions
        # frequencies i = 1..n_cos used in cos(pi * i * tau)
        self.register_buffer("freqs", torch.arange(1, n_cos + 1, dtype=torch.float32) * 3.141592653589793)

    def quantile_embedding(self, taus):
        # taus: (B, M) -> phi: (B, M, d)
        cos = torch.cos(taus.unsqueeze(-1) * self.freqs)   # (B, M, n_cos)
        return F.relu(self.phi(cos))                        # (B, M, d)

    def quantile_values(self, x, taus):
        # returns Z_{tau}(x, .) for each of M sampled taus: (B, M, n_actions)
        psi = self.torso(x).unsqueeze(1)                   # (B, 1, d)
        phi = self.quantile_embedding(taus)                # (B, M, d)
        feats = psi * phi                                  # Hadamard modulation (B, M, d)
        return self.head(feats)                            # (B, M, n_actions)

    def greedy_action(self, x, k=K_POLICY):
        b = x.shape[0]
        taus = torch.rand(b, k, device=x.device)           # tau ~ U([0,1]) (use U([0,eta]) for CVaR)
        z = self.quantile_values(x, taus)                  # (B, K, n_actions)
        q = z.mean(dim=1)                                  # MC estimate of E_tau[Z_tau] = mean
        return q.argmax(dim=1)


def quantile_huber_loss(td, taus, kappa=KAPPA):
    # td: (B, N_i, N'_j) pairwise errors delta = Ttheta_j - theta_i; taus: (B, N_i) levels of predictions
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    weight = (taus.unsqueeze(-1) - (td.detach() < 0).float()).abs()   # |tau_i - 1{delta<0}|
    rho = weight * huber / kappa
    # average over target samples j (1/N'), sum over prediction quantiles i, mean over batch
    return rho.mean(dim=2).sum(dim=1).mean()


def iqn_loss(online_net, target_net, obs, actions, rewards, next_obs, dones, gamma,
             n=N_QUANTILES, n_tgt=N_QUANTILES_TGT):
    b = obs.shape[0]
    device = obs.device
    with torch.no_grad():
        a_star = target_net.greedy_action(next_obs)                     # mean-greedy on target net
        taus_tgt = torch.rand(b, n_tgt, device=device)                  # tau' ~ U([0,1])
        z_next = target_net.quantile_values(next_obs, taus_tgt)         # (B, N', n_actions)
        z_next = z_next.gather(2, a_star.view(b, 1, 1).expand(b, n_tgt, 1)).squeeze(2)  # (B, N')
        not_done = (1.0 - dones.float()).view(b, 1)
        Tz = rewards.view(b, 1) + gamma * not_done * z_next            # (B, N') target locations
    taus = torch.rand(b, n, device=device)                             # tau ~ U([0,1])
    z = online_net.quantile_values(obs, taus)                          # (B, N, n_actions)
    theta = z.gather(2, actions.view(b, 1, 1).expand(b, n, 1)).squeeze(2)  # (B, N) predicted
    td = Tz.unsqueeze(1) - theta.unsqueeze(2)                          # (B, N, N'): delta_ij
    return quantile_huber_loss(td, taus)
```
