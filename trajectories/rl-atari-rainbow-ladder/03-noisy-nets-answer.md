# Noisy Nets — learned parametric exploration noise

**Problem.** $\epsilon$-greedy explores by re-flipping a coin at the action output every step: noise that
is local in time (independent each step, so no coherent multi-step plan forms) and uniform across states.
On games whose reward requires a $k$-step deliberate detour, the chance of stumbling through it decays like
$\epsilon^k$ — it essentially never happens. And $\epsilon$ is a hand-set schedule, not learned.

**Key idea.** Move the stochasticity upstream into the *parameters*. A noisy parameter is
$\theta=\mu+\sigma\odot\epsilon$ with $\epsilon$ zero-mean fixed-statistics noise and $\mu,\sigma$
learnable. A held-fixed draw induces a *different value function*; because the perturbation flows through
the net its effect is state-dependent, giving structured, temporally-extended exploration $\epsilon$-greedy
cannot. Minimizing $\bar L=\mathbb{E}_\epsilon[L(\mu+\sigma\odot\epsilon)]$ by reparameterization makes the
$\sigma$-gradient the local weight-gradient times $\epsilon$, so backprop *learns the noise scale per
parameter* — up where exploration still helps, toward zero where behavior has settled.

**Mechanics.** Replace the head's FC layers with **noisy linear** layers,
$y=(\mu^w+\sigma^w\odot\epsilon^w)x+(\mu^b+\sigma^b\odot\epsilon^b)$. **Factorized** noise:
$\epsilon^w_{j,i}=f(\epsilon^{\text{out}}_j)f(\epsilon^{\text{in}}_i)$,
$\epsilon^b_j=f(\epsilon^{\text{out}}_j)$, $f(x)=\operatorname{sign}(x)\sqrt{|x|}$ — $p+q$ draws instead of
$pq+q$. Delete $\epsilon$-greedy; act greedily under the sampled net. Resample between optimization steps;
online and target nets get *independent* draws (so the target is not correlated with the estimate). The
Double-DQN target is kept: online sampled net selects, target sampled net evaluates.

**What this moves.** The gain is concentrated on the minority of hard-exploration games with long
reward-free corridors; on the bulk of the suite, where Double DQN already explores enough, little changes.
Since the median is a robustness statistic, expect it to clear 117% only slightly.

```python
# Noisy Nets: factorized NoisyLinear heads replace epsilon-greedy. Greedy acting under the sampled net.
# Code home: vwxyzjn/cleanrl; excerpted from methods/noisy-nets/results/answer.md.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
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
        return x.sign().mul_(x.abs().sqrt_())                       # f(x) = sign(x) sqrt(|x|)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))             # outer product: p+q draws
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)


class NoisyDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.fc = NoisyLinear(3136, 512)
        self.head = NoisyLinear(512, n_actions)

    def forward(self, x):
        h = F.relu(self.fc(self.torso(x / 255.0)))
        return self.head(h)

    def reset_noise(self):
        self.fc.reset_noise()
        self.head.reset_noise()


def act(net, x):
    net.reset_noise()
    with torch.no_grad():
        return net(x).argmax(dim=1)                                # greedy under the sampled net


def noisy_double_dqn_target(online, target, rewards, next_obs, dones, gamma):
    online.reset_noise(); target.reset_noise()                    # INDEPENDENT draws online vs target
    with torch.no_grad():
        next_actions = online(next_obs).argmax(dim=1)             # select with online sample
        target_q = target(next_obs).gather(1, next_actions[:, None]).squeeze(1)  # eval with target sample
        return rewards + gamma * target_q * (1.0 - dones)
```
