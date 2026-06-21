Off-policy actor-critic methods for continuous control all inherit the same fixture from DQN: a target network. DDPG, TD3, and SAC regress the live critic toward r + γ V(s') where V(s') is read from a slowly tracked copy Q_{θ̄} with θ̄ ← (1−τ)θ̄ + τθ. The received wisdom is that this is necessary because bootstrapping a deep network off itself otherwise diverges: the target moves every gradient step, the regression objective is non-stationary, and residual error compounds. The cost is a deliberately stale value signal and another hyperparameter to tune. High-UTD methods such as REDQ and DroQ do not question the target network; they accept it and buy sample efficiency with large randomized ensembles or many critic updates per environment step. I want to ask whether the target network is load-bearing, or whether it is papering over a different problem.

The real issue is a distribution mismatch between the two input clouds that the same critic must evaluate. The current batch (s, a) comes from the replay buffer and contains actions sampled from older policies. The next batch (s', a') comes from the current policy. These are different state and action distributions, yet the Bellman equation asks the critic to relate Q(s, a) to Q(s', a') as if they were outputs of the same function. Without a strong constraint, the critic can drift to different scales on the two clouds, and the bootstrap amplifies the mismatch. The target network mitigates this by freezing one side, but it does not solve it; it only slows the drift and leaves the staleness behind.

The method I propose is CrossQ (Bhatt, Palenicek et al., ICLR 2024). It removes the target network entirely by making the critic normalize both input clouds under a single shared distribution. The critic uses Batch Renormalization, which is BatchNorm augmented with clipped correction terms r and d that tie batch statistics to running statistics. This matters because RL data is non-stationary: the policy changes, the replay distribution drifts, and batch statistics can swing. Batch Renormalization keeps training-time and running-time normalization consistent, which is the robustness plain BatchNorm lacks here.

The defining operation is the joint forward pass. Instead of forwarding (s, a) and (s', a') through the critic in two separate passes, CrossQ concatenates them into one batch of size 2N and runs a single forward pass through the BatchRenorm critic, then splits the output back into current and next halves. Because BatchRenorm computes its normalization moments from the union of both sub-batches, every input — whether current or next — is normalized identically. The prediction and the bootstrap value therefore live on the same normalized scale by construction. The next-state half is detached before the target is formed, so there is no degenerate gradient path through the bootstrap; but it is produced by the same normalized function as the prediction. This shared normalization supplies both the stationarity and the cross-batch consistency that the target network was approximating, so the target network can be deleted.

Everything else stays from SAC. The actor remains a stochastic tanh-Gaussian policy trained by reparameterization with the log-prob correction for the tanh squashing. Twin critics are kept, and the next-state target uses their elementwise minimum before subtracting the entropy bonus. The entropy temperature α is learned automatically by descending −log α · (log π + H_target) with H_target = −dim(A). Because the critic now learns from a live, un-stale bootstrap and the normalization keeps it stable, the actor can be updated frequently and the update-to-data ratio can stay at 1. The paper also widens the critic to 2048 hidden units, which the normalization makes trainable.

One subtlety is why naive BatchNorm fails in RL and why the joint pass fixes it. With separate passes, BatchNorm normalizes the current batch by its own moments and the next batch by its own moments. The critic then becomes two different functions — same weights, different normalization — and the Bellman equation relates outputs of two non-identical functions. Running statistics also lurch because they are updated separately by each pass. The joint pass forces one shared normalization, so the critic is one consistent function across both inputs and the running statistics follow one stable population. During the actor update, the critic is put in eval mode and evaluated only on the current-state batch, so it uses running statistics rather than recomputing batch statistics on a single distribution. In short, CrossQ answers the target-network question by attacking the distribution mismatch the target network was hiding, giving a target-free algorithm at UTD = 1 with only a few lines added to SAC.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class BatchRenorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=1e-3, rmax=3.0, dmax=5.0):
        super().__init__()
        self.momentum, self.eps, self.rmax, self.dmax = momentum, eps, rmax, dmax
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            bmean = x.mean(0)
            bvar = x.var(0, unbiased=False)
            bstd = (bvar + self.eps).sqrt()
            rstd = (self.running_var + self.eps).sqrt()
            r = (bstd / rstd).detach().clamp(1.0 / self.rmax, self.rmax)
            d = ((bmean - self.running_mean) / rstd).detach().clamp(-self.dmax, self.dmax)
            xhat = (x - bmean) / bstd * r + d
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * bmean.detach())
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * bvar.detach())
        else:
            xhat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        return self.weight * xhat + self.bias


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=2048):
        super().__init__()
        self.l1 = nn.Linear(obs_dim + act_dim, hidden)
        self.bn1 = BatchRenorm1d(hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.bn2 = BatchRenorm1d(hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        return self.l3(x).view(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action, hidden=256):
        super().__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        log_std = torch.tanh(self.log_std(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return self.mu(x), log_std

    def sample(self, s):
        mu, log_std = self(s)
        normal = torch.distributions.Normal(mu, log_std.exp())
        u = normal.rsample()
        y = torch.tanh(u)
        a = y * self.max_action
        logp = normal.log_prob(u) - torch.log(self.max_action * (1 - y.pow(2)) + 1e-6)
        return a, logp.sum(1, keepdim=True)


class CrossQ:
    def __init__(self, obs_dim, act_dim, max_action, device,
                 gamma=0.99, lr=1e-3, policy_delay=3):
        self.device, self.gamma, self.policy_delay = device, gamma, policy_delay
        self.max_action, self.total_it = max_action, 0
        self.actor = Actor(obs_dim, act_dim, max_action).to(device)
        self.qf1 = Critic(obs_dim, act_dim).to(device)
        self.qf2 = Critic(obs_dim, act_dim).to(device)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr)
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.al_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    def _joint(self, qf, s, a, s2, a2):
        q = qf(torch.cat([s, s2], 0), torch.cat([a, a2], 0))
        return q[: s.shape[0]], q[s.shape[0] :]

    def update(self, batch):
        self.total_it += 1
        s, s2, a, r, d = batch
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
        self.qf1.train(); self.qf2.train()
        q1_cur, q1_nxt = self._joint(self.qf1, s, a, s2, a2)
        q2_cur, q2_nxt = self._joint(self.qf2, s, a, s2, a2)
        with torch.no_grad():
            min_nxt = torch.min(q1_nxt.detach(), q2_nxt.detach()) - self.alpha * logp2.view(-1)
            target = r + (1 - d) * self.gamma * min_nxt
        q_loss = F.mse_loss(q1_cur, target) + F.mse_loss(q2_cur, target)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        if self.total_it % self.policy_delay == 0:
            self.qf1.eval(); self.qf2.eval()
            pi, logp = self.actor.sample(s)
            min_pi = torch.min(self.qf1(s, pi), self.qf2(s, pi))
            a_loss = (self.alpha * logp.view(-1) - min_pi).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            alpha_loss = (-self.log_alpha.exp() * (logp.detach().view(-1) + self.target_entropy)).mean()
            self.al_opt.zero_grad(); alpha_loss.backward(); self.al_opt.step()
            self.alpha = self.log_alpha.exp().item()
        return {"q_loss": q_loss.item()}
```
