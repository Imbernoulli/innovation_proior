# QR-DQN (Quantile Regression DQN)

**Problem.** The distributional Bellman operator $\mathcal{T}^\pi$ is a $\gamma$-contraction in the
maximal Wasserstein metric, but Wasserstein cannot be minimized from samples by SGD (biased
gradients). A fixed-atom categorical agent (C51) sidesteps this by learning probabilities on a
fixed support, projecting the Bellman target onto that support, and minimizing KL — so it does not
actually minimize Wasserstein and needs the support bounds $[V_{\min},V_{\max}]$ plus a projection.

**Key idea — transpose the parametrization.** Fix the probabilities to uniform $1/N$ and learn the
*locations*:
$$Z_\theta(x,a)=\frac1N\sum_{i=1}^N\delta_{\theta_i(x,a)}.$$
The $N$ equal-mass locations are *quantiles* of the return. This removes the support bounds and the
projection, and lets the locations be trained — via quantile regression — to hit the $W_1$ projection
with *unbiased* gradients.

**Which quantiles.** For ordered locations, minimizing
$W_1(Y,Z_\theta)=\sum_i\int_{\tau_{i-1}}^{\tau_i}|F_Y^{-1}(\omega)-\theta_i|\,d\omega$
($\tau_i=i/N$) cell-by-cell: the subgradient of each cell is $2F(\theta)-(\tau_{i-1}+\tau_i)$, zero
at $F(\theta)=\frac{\tau_{i-1}+\tau_i}{2}$. So the $W_1$-optimal locations are the **midpoint
quantiles**
$$\hat\tau_i=\frac{2i-1}{2N},\qquad \theta_i=F_Y^{-1}(\hat\tau_i).$$

**Quantile regression loss.** The $\tau$-quantile is the minimizer of $\mathbb{E}[\rho_\tau(\hat Z-\theta)]$
with
$$\rho_\tau(u)=u\,(\tau-\mathbb{1}_{u<0}),$$
whose error-gradient $\tau-\mathbb{1}_{u<0}$ depends only on the sign of $u$; the corresponding
stochastic gradient for $\theta$ is unbiased from one sample.
The location objective $\sum_i\mathbb{E}_{\hat Z}[\rho_{\hat\tau_i}(\hat Z-\theta_i)]$ is minimized
at the $W_1$ projection.

**Quantile Huber loss.** $\rho_\tau$ is non-smooth at $0$ (constant-magnitude gradient), which
hurts nonlinear function approximation. Huberize:
$$\mathcal{L}_\kappa(u)=\begin{cases}\frac12u^2,&|u|\le\kappa\\ \kappa(|u|-\frac12\kappa),&|u|>\kappa\end{cases},
\qquad \rho^\kappa_\tau(u)=\big|\tau-\mathbb{1}_{u<0}\big|\,\mathcal{L}_\kappa(u).$$
The hard-loss setting $\kappa=0$ is implemented as the separate branch
$\rho^0_\tau(u)=\rho_\tau(u)=|\tau-\mathbb{1}_{u<0}|\,|u|$.

**QRTD (policy evaluation).** $\theta_i(x)\leftarrow\theta_i(x)+\alpha(\hat\tau_i-\mathbb{1}\{r+\gamma z'<\theta_i(x)\})$,
$z'\sim Z_\theta(x')$; in practice average over all next-state locations.

**QR-DQN (control).** Distributional optimality operator with the next action greedy on the *mean*
of the next-state distribution, $a^\star=\arg\max_{a'}\frac1N\sum_j\theta_j(x',a')$. Three minimal
changes to DQN: output layer $|\mathcal{A}|\times N$; quantile Huber loss instead of Huber; Adam
instead of RMSProp. Per transition, target locations $\mathcal{T}\theta_j=r+\gamma\theta_j(x',a^\star)$
(from a target network, $\gamma=0$ at terminal), and the all-pairs loss
$$\mathcal{L}=\sum_{i=1}^N\mathbb{E}_j\big[\rho^\kappa_{\hat\tau_i}(\mathcal{T}\theta_j-\theta_i(x,a))\big].$$
Atari settings: $N=200$, $\kappa=1$ (QR-DQN-1) or $\kappa=0$ (QR-DQN-0), Adam $\alpha=5\times10^{-5}$,
$\epsilon_{\text{Adam}}=0.01/32$.

**Guarantee.** The projected operator $\Pi_{W_1}\mathcal{T}^\pi$ is a $\gamma$-contraction in the
$\infty$-Wasserstein metric $\bar d_\infty$ (unique fixed point $\hat Z^\pi$); since
$\bar d_p\le\bar d_\infty$, convergence holds for all $p$. It is *not* a contraction in $\bar d_p$
for $p<\infty$.

```python
import torch
import torch.nn as nn

N = 200
KAPPA = 1.0

class DistributionalQNetwork(nn.Module):
    def __init__(self, n_actions, n=N):
        super().__init__()
        self.n_actions, self.n = n_actions, n
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions * n))
        self.register_buffer("levels", (torch.arange(n, dtype=torch.float32) + 0.5) / n)  # midpoints

    def prediction(self, x):
        return self.net(x.float() / 255.0).view(-1, self.n_actions, self.n)   # (B, A, N)

    def greedy_action(self, x):
        return self.prediction(x).mean(dim=2).argmax(dim=1)                   # greedy on the mean

def bootstrap_target(target_net, rewards, next_obs, dones, gamma):
    with torch.no_grad():
        nq = target_net.prediction(next_obs)                                  # (B, A, N)
        a_star = nq.mean(dim=2).argmax(dim=1)
        batch = torch.arange(next_obs.size(0), device=next_obs.device)
        next_theta = nq[batch, a_star]                                        # (B, N)
        not_done = 1.0 - dones.float().view(-1, 1)
        return rewards.float().view(-1, 1) + gamma * not_done * next_theta    # (B, N); no projection

def pairwise_distribution_loss(theta, target, levels, kappa=KAPPA):
    u = target.unsqueeze(1) - theta.unsqueeze(2)                              # u_ij = Ttheta_j - theta_i
    tau = levels.view(1, -1, 1)
    weight = (tau - (u.detach() < 0).float()).abs()
    if kappa == 0:
        rho = weight * u.abs()
    else:
        abs_u = u.abs()
        huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
        rho = weight * huber                                                  # |tau_i - 1{u<0}| L_kappa
    return rho.mean(dim=2).sum(dim=1).mean()                                  # sum_i E_j[rho], batch-mean

def loss_fn(online_net, obs, actions, target, kappa=KAPPA):
    pred = online_net.prediction(obs)
    batch = torch.arange(obs.size(0), device=obs.device)
    theta = pred[batch, actions]                                              # (B, N)
    return pairwise_distribution_loss(theta, target, online_net.levels, kappa)
```
