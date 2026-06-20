# Rainbow — all six combined (+ multi-step), one agent

**Premise.** Six improvements were each measured in isolation against the DQN floor; the best single one
(distributional, 164%) is the bar. They touch *largely different* surfaces of the agent — target
construction, sampling, architecture, exploration, representation — so they should *compound* rather than
collide, provided the three specified for a scalar are re-expressed over the now-distributional value.

**Integration (the design, not a bolt-on).** Make the categorical return distribution the central object.
- **Double Q-learning over the distribution:** pick $a^\star=\arg\max_a z^\top p_\theta(S_{t+n},a)$ with
  the *online* net's mean, bootstrap the *target* net's distribution $p_{\bar\theta}(S_{t+n},a^\star)$.
- **Dueling over the logits:** per-atom value + advantage streams, mean-subtracted on the logits, then
  softmax over atoms — same head, distributional output.
- **Prioritized replay over the distribution:** priority $\propto$ the per-sample KL/cross-entropy loss
  $L_t^\omega$ (the more principled surprise signal), not a scalar TD error.
- **Noisy Nets:** noisy linear layers replace the head's FC layers; act greedily with $\epsilon=0$.
- **Distributional RL:** project shifted/scaled atoms back to the fixed 51-atom $[-10,10]$ grid, minimize
  $D_{\mathrm{KL}}(\Phi_z d_t^{(n)}\|d_t)$.
- **Multi-step (added because it composes for free):** shift target atoms by $R_t^{(n)}$, contract by
  $\gamma_t^{(n)}=\gamma^n$ — the C51 shift-and-scale with $r\to R_t^{(n)}$, $\gamma\to\gamma^n$; $n=3$.

**Integrated target and loss.**
$$d_t^{(n)}=\Big(R_t^{(n)}+\gamma_t^{(n)}z,\;p_{\bar\theta}\big(S_{t+n},\arg\max_a z^\top p_\theta(S_{t+n},a)\big)\Big),\qquad
L_t=-\sum_i(\Phi_z d_t^{(n)})_i\log p^i_\theta(S_t,A_t).$$
Replay priorities $\propto L_t^\omega$; importance weights multiply the minibatch loss but are not folded
into the stored priority.

**Bar.** Floor 79%, best single component 164%. If the six were redundant the combined agent would not
clear $\approx164\%$ (the null); if largely independent it should clear it decisively and roughly double
the floor into the low 200s — the best value-based learner on the suite.

```python
# Rainbow: distributional + dueling-over-logits + double + noisy + prioritized + n-step, one agent.
# Code home: vwxyzjn/cleanrl + Kaixhin/Rainbow; excerpted from methods/rainbow/results/answer.md.
import math
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
        self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.register_buffer("weight_epsilon", torch.empty(out_f, in_f))
        self.bias_mu = nn.Parameter(torch.empty(out_f))
        self.bias_sigma = nn.Parameter(torch.empty(out_f))
        self.register_buffer("bias_epsilon", torch.empty(out_f))
        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_f)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_f))
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_f))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_f)
        eps_out = self._scale_noise(self.out_f)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, w, b)
        return F.linear(x, self.weight_mu, self.bias_mu)


class RainbowNet(nn.Module):
    def __init__(self, n_actions, n_atoms=N_ATOMS):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.register_buffer("z", torch.linspace(V_MIN, V_MAX, n_atoms))
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.fc_h_v = NoisyLinear(3136, 512)       # noisy FC: exploration on the head weights
        self.fc_h_a = NoisyLinear(3136, 512)
        self.fc_z_v = NoisyLinear(512, n_atoms)
        self.fc_z_a = NoisyLinear(512, n_actions * n_atoms)

    def forward(self, x, log=False):
        phi = self.torso(x).view(-1, 3136)
        v = self.fc_z_v(F.relu(self.fc_h_v(phi))).view(-1, 1, self.n_atoms)
        a = self.fc_z_a(F.relu(self.fc_h_a(phi))).view(-1, self.n_actions, self.n_atoms)
        logits = v + a - a.mean(dim=1, keepdim=True)        # DUELING per atom, on the logits
        return F.log_softmax(logits, dim=2) if log else F.softmax(logits, dim=2)

    def reset_noise(self):
        for m in (self.fc_h_v, self.fc_h_a, self.fc_z_v, self.fc_z_a):
            m.reset_noise()


def act(net, x):
    with torch.no_grad():
        probs = net(x)
        return (probs * net.z).sum(dim=2).argmax(dim=1)     # mean-greedy, epsilon = 0


def learn(online, target, obs, actions, n_returns, next_obs, nonterminal, gamma, weights):
    batch = actions.numel()
    arange = torch.arange(batch, device=actions.device)
    z = online.z
    delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)

    log_ps_a = online(obs, log=True)[arange, actions]

    with torch.no_grad():
        a_star = (online(next_obs) * z).sum(dim=2).argmax(dim=1)   # DOUBLE: select by online mean
        target.reset_noise()
        next_target = target(next_obs)[arange, a_star]             # ... evaluate target distribution
        Tz = n_returns[:, None] + nonterminal.view(-1, 1) * (gamma ** N_STEP) * z[None, :]  # n-STEP shift
        Tz = Tz.clamp(V_MIN, V_MAX)
        b = (Tz - V_MIN) / delta_z
        lower, upper = b.floor().long(), b.ceil().long()
        lower[(upper > 0) & (lower == upper)] -= 1
        upper[(lower < N_ATOMS - 1) & (lower == upper)] += 1
        m = torch.zeros(batch, N_ATOMS, device=obs.device)        # project Phi onto the fixed grid
        offset = (torch.arange(batch, device=obs.device) * N_ATOMS).unsqueeze(1)
        m.view(-1).index_add_(0, (lower + offset).reshape(-1),
                              (next_target * (upper.float() - b)).reshape(-1))
        m.view(-1).index_add_(0, (upper + offset).reshape(-1),
                              (next_target * (b - lower.float())).reshape(-1))

    per_sample_loss = -(m * log_ps_a).sum(dim=1)                   # cross-entropy of KL
    loss = (weights * per_sample_loss).mean()                     # IS weights on the gradient
    priorities = per_sample_loss.detach().cpu().numpy()           # PRIORITIZED by the distributional loss
    return loss, priorities
```

Default Atari hyperparameters: minimum history 80K frames; Adam learning rate $6.25\times10^{-5}$;
Adam $\epsilon=1.5\times10^{-4}$; target update period 32K frames; proportional priority exponent
$\omega=0.5$; importance-sampling exponent $\beta:0.4\to1.0$; $n=3$; 51 atoms; support $[-10,10]$;
noisy-layer $\sigma_0=0.5$; $\epsilon=0$.
