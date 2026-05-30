# Rainbow

**Problem.** Six independent DQN improvements each fix a distinct limitation and each help in
isolation, but it is unknown whether they are complementary or how to integrate them — several
modify the same machinery (the bootstrap target, the loss, the head, the replay distribution, the
action rule). Rainbow integrates all six into a single agent and ablates each.

**The six components and what each fixes.**
- **Double Q-learning** — overestimation of the bootstrap $\max$: select with the online net,
  evaluate with the target net.
- **Prioritized replay** — wasted updates on learned transitions: sample $\propto|\text{error}|^\omega$,
  IS-correct with $\beta$.
- **Dueling networks** — value/action conflation:
  $q=v+a-\frac1{N_a}\sum_{a'}a$ (mean-subtraction fixes identifiability).
- **Multi-step** — slow reward propagation / high bootstrap bias: $n$-step return
  $R_t^{(n)}=\sum_{k=0}^{n-1}\gamma_t^{(k)}R_{t+k+1}$.
- **Distributional (C51)** — mean-only values: categorical distribution on a fixed atom support,
  projected KL loss.
- **Noisy Nets** — shallow exploration: learnable per-weight noise; act greedily, $\epsilon=0$.

**Integration (the contribution).** Take the categorical distribution as the spine.
- *Multi-step distributional target*: contract the atoms by the cumulative discount and shift by
  the $n$-step return,
  $d_t^{(n)}=(R_t^{(n)}+\gamma_t^{(n)}\bm z,\ \bm p_{\bar\theta}(S_{t+n},a^*_{t+n}))$,
  loss $D_{\text{KL}}(\Phi_{\bm z}d_t^{(n)}\,\|\,d_t)$.
- *Double Q*: $a^*_{t+n}=\arg\max_a \bm z^\top\bm p_\theta(S_{t+n},a)$ selected by the **online**
  net (on the mean), distribution read from the **target** net.
- *Prioritized replay by KL*: $p_t\propto(D_{\text{KL}}(\Phi_{\bm z}d_t^{(n)}\,\|\,d_t))^\omega$ —
  the quantity being minimized; more robust to stochastic returns than $|\text{TD}|$.
- *Dueling, per atom*:
  $p^i_\theta(s,a)=\mathrm{softmax}_i\big(v^i_\eta(\phi)+a^i_\psi(\phi,a)-\tfrac1{N_a}\sum_{a'}a^i_\psi(\phi,a')\big)$,
  $\phi=f_\xi(s)$ — aggregate value/advantage on each atom's logit, then softmax over atoms.
- *Noisy Nets*: replace **all** linear layers with factorised-Gaussian noisy layers; drop
  $\epsilon$-greedy.

**Hyperparameters (identical across all 57 games).** Min history 80K frames; Adam
$\alpha=6.25\times10^{-5}$ (DQN's $/4$), Adam $\epsilon=1.5\times10^{-4}$; $\epsilon=0$; noisy
$\sigma_0=0.5$; target period 32K frames; proportional prioritization $\omega=0.5$, IS
$\beta:0.4\to1.0$; $n=3$; 51 atoms; $[v_{\min},v_{\max}]=[-10,10]$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ATOMS = 51
V_MIN, V_MAX = -10.0, 10.0
N_STEP = 3

class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma0=0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.w_mu = nn.Parameter(torch.empty(out_f, in_f)); self.w_sig = nn.Parameter(torch.empty(out_f, in_f))
        self.register_buffer("w_eps", torch.empty(out_f, in_f))
        self.b_mu = nn.Parameter(torch.empty(out_f)); self.b_sig = nn.Parameter(torch.empty(out_f))
        self.register_buffer("b_eps", torch.empty(out_f))
        mu = 1.0 / in_f ** 0.5
        self.w_mu.data.uniform_(-mu, mu); self.b_mu.data.uniform_(-mu, mu)
        self.w_sig.data.fill_(sigma0 / in_f ** 0.5); self.b_sig.data.fill_(sigma0 / out_f ** 0.5)
        self.reset_noise()
    def _f(self, n):
        x = torch.randn(n); return x.sign() * x.abs().sqrt()
    def reset_noise(self):
        ei, eo = self._f(self.in_f), self._f(self.out_f)
        self.w_eps.copy_(eo.ger(ei)); self.b_eps.copy_(eo)
    def forward(self, x):
        if self.training:
            return F.linear(x, self.w_mu + self.w_sig * self.w_eps, self.b_mu + self.b_sig * self.b_eps)
        return F.linear(x, self.w_mu, self.b_mu)

class RainbowNet(nn.Module):
    def __init__(self, n_actions, n_atoms=N_ATOMS):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.register_buffer("z", torch.linspace(V_MIN, V_MAX, n_atoms))
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.fc_v = NoisyLinear(3136, 512); self.v = NoisyLinear(512, n_atoms)
        self.fc_a = NoisyLinear(3136, 512); self.a = NoisyLinear(512, n_actions * n_atoms)
    def dist(self, x):
        phi = self.torso(x / 255.0)
        v = self.v(F.relu(self.fc_v(phi))).view(-1, 1, self.n_atoms)
        a = self.a(F.relu(self.fc_a(phi))).view(-1, self.n_actions, self.n_atoms)
        logits = v + a - a.mean(dim=1, keepdim=True)          # dueling aggregation per atom
        return F.softmax(logits, dim=2)
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

def act(net, x):
    with torch.no_grad():
        p = net.dist(x); return (p * net.z).sum(2).argmax(1)  # greedy on the mean; noise explores

def learn(online, target, obs, actions, n_returns, next_obs, nonterminal, gamma, weights):
    online.reset_noise(); target.reset_noise()
    z = online.z; dz = (V_MAX - V_MIN) / (N_ATOMS - 1)
    with torch.no_grad():
        a_star = (online.dist(next_obs) * z).sum(2).argmax(1)             # double-Q select (online)
        pns = target.dist(next_obs)[torch.arange(len(next_obs)), a_star]  # evaluate (target)
        Tz = (n_returns[:, None] + nonterminal[:, None] * (gamma ** N_STEP) * z).clamp(V_MIN, V_MAX)
        b = (Tz - V_MIN) / dz; l, u = b.floor().long(), b.ceil().long()
        l[(u > 0) & (l == u)] -= 1; u[(l < N_ATOMS - 1) & (l == u)] += 1
        m = torch.zeros_like(pns)
        for i in range(m.size(0)):
            m[i].index_add_(0, l[i], pns[i] * (u[i].float() - b[i]))
            m[i].index_add_(0, u[i], pns[i] * (b[i] - l[i].float()))
    log_p = torch.log(online.dist(obs)[torch.arange(len(obs)), actions].clamp_min(1e-8))
    kl = -(m * log_p).sum(1)                                              # KL up to a constant
    loss = (weights * kl).mean()
    return loss, kl.detach().abs()                                       # priority = KL loss
```
