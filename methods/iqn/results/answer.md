# IQN (Implicit Quantile Networks)

**Problem.** Distributional value-based RL helps even when the policy acts on the mean, but every prior agent caps the resolution of the learned return distribution with an architectural constant: C51 fixes $N$ atoms on a prescribed $[V_{\min},V_{\max}]$; QR-DQN fixes $N$ quantile locations at predetermined levels $\hat\tau_i=(2i-1)/(2N)$ and emits an $|\mathcal A|\times N$ output. Fidelity is bounded by a hand-set $N$, not by network capacity or training, and the policy reads only the mean.

**Key idea — learn the quantile function implicitly.** Model the inverse CDF directly as a network input: for a base sample $\tau\sim U([0,1])$, $Z_\tau(x,a):=F^{-1}_{Z(x,a)}(\tau)$ is a sample of the return. The distribution is defined implicitly by reparameterizing the uniform base sample through the learned quantile map, so its resolution is bounded only by capacity and training. The number of quantiles is no longer an output-layer width — it is a *sample count per update*, chosen freely.

**Architecture.** Keep the DQN state encoder $\psi(x)$ and the action head $f$. In the fixed Nature-DQN re-expression used below, $\psi$ includes the convolutional stack plus the usual 512-dimensional hidden layer, so the Hadamard product happens at $d=512$ and $f$ is a single `Linear(512, n_actions)` head. Add a sample embedding $\phi(\tau)$ and combine multiplicatively so that shallow head still sees a state-dependent $\tau$ interaction:
$$Z_\tau(x,a)\approx f\big(\psi(x)\odot\phi(\tau)\big)_a,\qquad
\phi_j(\tau)=\operatorname{ReLU}\!\Big(\sum_{i=1}^{n}\cos(\pi i\tau)\,w_{ij}+b_j\Big),\ n=64,$$
where $\odot$ is the Hadamard product. Indexing the cosine basis from $0$ instead only adds a constant term that can be absorbed into $b_j$; the implementation convention below uses $i=1,\ldots,n$. Concatenation would make $\tau$ enter only additively through the shallow $f$ (a state-independent shift of the curve); multiplication forces feature-wise modulation.

**Loss.** Two independent base samples per term give the sampled TD error
$$\delta^{\tau,\tau'}_t=r_t+\gamma\,Z_{\tau'}\big(x_{t+1},\pi(x_{t+1})\big)-Z_{\tau}(x_t,a_t),$$
and the all-pairs quantile Huber loss over $N$ prediction samples $\tau_i$ and $N'$ target samples $\tau'_j$:
$$\mathcal L=\frac1{N'}\sum_{i=1}^{N}\sum_{j=1}^{N'}\rho^\kappa_{\tau_i}\big(\delta^{\tau_i,\tau'_j}_t\big),\qquad
\rho^\kappa_\tau(u)=\big|\tau-\mathbb{1}\{u<0\}\big|\,\frac{\mathcal L_\kappa(u)}{\kappa},$$
with
$$\mathcal L_\kappa(u)=\begin{cases}\frac12u^2,&|u|\le\kappa,\\ \kappa(|u|-\frac12\kappa),&|u|>\kappa.\end{cases}$$
Use $\kappa=1$. The quantile levels are sampled fresh each update (continuous, not a grid), so the function is supervised across changing probability levels. $N$ controls how many predicted levels are shaped per update; $N'$ averages over target samples. The default setting here is $N=N'=8$.

**Policy.** Mean-greedy by Monte Carlo: $\tilde\pi(x)=\argmax_a\frac1K\sum_{k=1}^K Z_{\tilde\tau_k}(x,a)$, $\tilde\tau_k\sim U([0,1])$, $K=32$. Replacing the uniform quantile-level sampling with a distribution $\mu$ gives the distorted expectation $\int_0^1 F^{-1}_Z(u)\,d\mu(u)$: mass near low quantiles is risk-averse, mass near high quantiles is risk-seeking. CVaR$(\eta)$ is the lower-tail case $u\sim U([0,\eta])$, estimating $\frac1\eta\int_0^\eta F^{-1}_Z(u)\,du$. The risk-neutral agent uses $\mu=U([0,1])$.

**Settings.** $n=64$ cosine embedding, $N=N'=8$, $K=32$, $\kappa=1$, Adam, the standard Atari DQN protocol (target network, $10^6$ uniform replay, $\epsilon$-greedy, $84\times84\times4$ frames).

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
