# Noisy Networks for Exploration

Replace action-space dithering with learnable parameter noise in the network heads.
For noisy parameters

```text
theta = mu + Sigma * epsilon
zeta = (mu, Sigma)
Lbar(zeta) = E_epsilon[L(mu + Sigma * epsilon)]
grad_zeta Lbar(zeta) ~= grad_{mu,Sigma} L(mu + Sigma * xi)
```

`epsilon` is zero-mean fixed-statistics noise; `mu` and `Sigma` are trained by ordinary
backpropagation. This is not a posterior approximation: it is a learnable exploration
mechanism.

For a linear layer with `p` inputs and `q` outputs:

```text
y = (mu_w + sigma_w * epsilon_w) x + mu_b + sigma_b * epsilon_b
```

where `mu_w, sigma_w` have shape `q x p` and `mu_b, sigma_b` have shape `q`.

Noise cases:

- Independent Gaussian: every `epsilon_w[j,i]` and `epsilon_b[j]` is iid `N(0,1)`;
  this costs `pq + q` draws and is the paper's main A3C choice.
- Factorised Gaussian: draw `epsilon_in[i]` for inputs and `epsilon_out[j]` for outputs,
  use `f(x) = sign(x) sqrt(abs(x))`, then
  `epsilon_w[j,i] = f(epsilon_out[j]) f(epsilon_in[i])` and
  `epsilon_b[j] = f(epsilon_out[j])`. This costs `p + q` draws and is used for DQN/Dueling.

Initialization cases:

- Independent: `mu ~ U[-sqrt(3/p), sqrt(3/p)]`, `sigma = 0.017`.
- Factorised paper rule: `mu ~ U[-1/sqrt(p), 1/sqrt(p)]`, `sigma = sigma_0/sqrt(p)`,
  `sigma_0 = 0.5`.
- Saved Rainbow reference code initializes `weight_sigma` with `sqrt(in_features)` and
  `bias_sigma` with `sqrt(out_features)`; the code below follows that reference convention.

RL integration:

- DQN/Dueling: remove `epsilon`-greedy and act greedily under the sampled value network.
  Resample after every replay/optimization step; because these agents update once per action,
  this means resampling before each action. Hold one sample fixed across a replay batch.
- Noisy DQN target:
  `r + gamma max_b Q(y,b,epsilon'; zeta^-)`, with independent online noise `epsilon` and
  target noise `epsilon'`.
- Noisy Dueling target:
  `b*(y) = argmax_b Q(y,b,epsilon''; zeta)` selected by the online network, evaluated as
  `Q(y,b*(y),epsilon'; zeta^-)` by the target network.
- A3C: remove the entropy bonus, make the policy/value head fully connected layers noisy,
  and keep the same noise sample fixed for the whole `n`-step rollout.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        return F.linear(input, self.weight_mu, self.bias_mu)


class NoisyDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = NoisyLinear(3136, 512)
        self.head = NoisyLinear(512, n_actions)

    def forward(self, x):
        h = self.torso(x.float() / 255.0)
        h = F.relu(self.fc(h))
        return self.head(h)

    def reset_noise(self):
        self.fc.reset_noise()
        self.head.reset_noise()


def act(net, x):
    net.reset_noise()
    with torch.no_grad():
        return net(x).argmax(dim=1)


def dqn_loss(online, target, obs, actions, rewards, next_obs, dones, gamma):
    online.reset_noise()
    target.reset_noise()
    q = online(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        target_q = target(next_obs).max(dim=1).values
        y = rewards + gamma * target_q * (1.0 - dones)
    return F.mse_loss(q, y)


def dueling_target(online, target, rewards, next_obs, dones, gamma):
    online.reset_noise()
    target.reset_noise()
    with torch.no_grad():
        next_actions = online(next_obs).argmax(dim=1)
        target_q = target(next_obs).gather(1, next_actions[:, None]).squeeze(1)
        return rewards + gamma * target_q * (1.0 - dones)
```
