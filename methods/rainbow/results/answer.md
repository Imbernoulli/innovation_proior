# Rainbow

Rainbow is a single DQN-family agent that combines six prior improvements by making the categorical
return distribution the central object.

- Double Q-learning: choose the bootstrap action with the online-network mean
  $z^\top p_\theta(S_{t+n},a)$ and evaluate its distribution with the target network.
- Prioritized replay: use the per-sample categorical KL/cross-entropy loss as the priority source,
  not a mean TD error.
- Dueling networks: aggregate value and advantage per atom on logits, then softmax over atoms for
  each action.
- Multi-step learning: shift target atoms by $R_t^{(n)}$ and contract them by $\gamma_t^{(n)}$.
- Distributional RL: project the shifted atoms back to the fixed 51-atom support $[-10,10]$ and
  minimize $D_{\mathrm{KL}}(\Phi_zd_t^{(n)}\|d_t)$.
- Noisy Nets: replace the fully connected layers with factorized noisy linear layers and act
  greedily with $\epsilon=0$.

The integrated target is

$$
d_t^{(n)}=
\left(R_t^{(n)}+\gamma_t^{(n)}z,\,
p_{\bar\theta}\left(S_{t+n},
\arg\max_a z^\top p_\theta(S_{t+n},a)\right)\right),
$$

and the optimized per-sample loss is the cross-entropy form of the KL,

$$
L_t=-\sum_i(\Phi_zd_t^{(n)})_i\log p^i_\theta(S_t,A_t).
$$

Replay priorities are proportional to $L_t^\omega$; importance weights multiply the minibatch loss
for the gradient but are not folded into the stored priority.

```python
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
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
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
        self.fc_h_v = NoisyLinear(3136, 512)
        self.fc_h_a = NoisyLinear(3136, 512)
        self.fc_z_v = NoisyLinear(512, n_atoms)
        self.fc_z_a = NoisyLinear(512, n_actions * n_atoms)

    def forward(self, x, log=False):
        # Match the reference layout: replay provides float frames already scaled to [0, 1].
        phi = self.torso(x).view(-1, 3136)
        v = self.fc_z_v(F.relu(self.fc_h_v(phi))).view(-1, 1, self.n_atoms)
        a = self.fc_z_a(F.relu(self.fc_h_a(phi))).view(-1, self.n_actions, self.n_atoms)
        logits = v + a - a.mean(dim=1, keepdim=True)
        return F.log_softmax(logits, dim=2) if log else F.softmax(logits, dim=2)

    def reset_noise(self):
        for module in (self.fc_h_v, self.fc_h_a, self.fc_z_v, self.fc_z_a):
            module.reset_noise()


def act(net, x):
    with torch.no_grad():
        probs = net(x)
        return (probs * net.z).sum(dim=2).argmax(dim=1)


def learn(online, target, obs, actions, n_returns, next_obs, nonterminal, gamma, weights):
    # The reference training loop samples online noise before acting/updating.
    batch = actions.numel()
    arange = torch.arange(batch, device=actions.device)
    z = online.z
    delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)

    log_ps_a = online(obs, log=True)[arange, actions]

    with torch.no_grad():
        next_online = online(next_obs)
        a_star = (next_online * z).sum(dim=2).argmax(dim=1)

        target.reset_noise()
        next_target = target(next_obs)[arange, a_star]

        Tz = n_returns[:, None] + nonterminal.view(-1, 1) * (gamma ** N_STEP) * z[None, :]
        Tz = Tz.clamp(V_MIN, V_MAX)
        b = (Tz - V_MIN) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()

        lower[(upper > 0) & (lower == upper)] -= 1
        upper[(lower < N_ATOMS - 1) & (lower == upper)] += 1

        m = torch.zeros(batch, N_ATOMS, device=obs.device)
        offset = (torch.arange(batch, device=obs.device) * N_ATOMS).unsqueeze(1)
        m.view(-1).index_add_(
            0, (lower + offset).reshape(-1),
            (next_target * (upper.float() - b)).reshape(-1))
        m.view(-1).index_add_(
            0, (upper + offset).reshape(-1),
            (next_target * (b - lower.float())).reshape(-1))

    per_sample_loss = -(m * log_ps_a).sum(dim=1)
    loss = (weights * per_sample_loss).mean()
    priorities = per_sample_loss.detach().cpu().numpy()
    return loss, priorities
```

Default Atari hyperparameters: minimum history 80K frames; Adam learning rate
$6.25\times10^{-5}$; Adam $\epsilon=1.5\times10^{-4}$; target update period 32K frames;
proportional priority exponent $\omega=0.5$; importance-sampling exponent $\beta:0.4\to1.0$;
$n=3$; 51 atoms; support $[-10,10]$; noisy-layer $\sigma_0=0.5$; $\epsilon=0$.
