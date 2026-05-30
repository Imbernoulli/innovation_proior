# NoisyNet

**Problem.** Deep-RL agents explore with *dithering*: $\epsilon$-greedy (value methods) or an
entropy bonus (policy gradient) inject per-step, temporally-decorrelated, mostly state-independent
noise into the action. This cannot produce the coherent multi-step deviations needed for "deep"
exploration, and the exploration magnitude is a hand-tuned hyperparameter held fixed across tasks.

**Key idea.** Move the noise from the *action* to the *weights*, and make its scale *learnable*.
Write each network parameter as
$$\theta = \mu + \Sigma\odot\varepsilon,$$
with learnable mean $\mu$ and learnable per-parameter noise scale $\Sigma$, and fixed-statistics
zero-mean noise $\varepsilon$. A single weight perturbation, held fixed over an episode, induces a
*consistent, state-dependent* change in the policy (it flows through the network, so the action
shift depends on the input). The expected loss $\bar L(\zeta)=\mathbb{E}_\varepsilon[L(\mu+\Sigma\odot\varepsilon)]$
is differentiable in $\zeta=(\mu,\Sigma)$ via the reparameterisation trick, with a single-sample
Monte-Carlo gradient $\nabla_\zeta\bar L\approx\nabla_{\mu,\Sigma}L(\mu+\Sigma\odot\xi)$. The
gradient on $\Sigma$ lets the RL loss *learn how much to explore*, per weight, self-annealing.

**Noisy linear layer.** For $p$ inputs, $q$ outputs:
$$y=(\mu^w+\sigma^w\odot\varepsilon^w)\,x+\mu^b+\sigma^b\odot\varepsilon^b,$$
learnables $\mu^w,\sigma^w\in\mathbb{R}^{q\times p}$, $\mu^b,\sigma^b\in\mathbb{R}^q$.

**Two noise schemes.**
- *Independent Gaussian*: every $\varepsilon^w_{i,j},\varepsilon^b_j\sim\mathcal{N}(0,1)$
  ($pq+q$ draws). Used for A3C.
- *Factorised Gaussian*: $p$ input draws $\varepsilon_i$ and $q$ output draws $\varepsilon_j$
  ($p+q$ total), with
  $$\varepsilon^w_{i,j}=f(\varepsilon_i)f(\varepsilon_j),\quad \varepsilon^b_j=f(\varepsilon_j),
  \quad f(x)=\operatorname{sgn}(x)\sqrt{|x|}.$$
  Cuts RNG cost; the square-root keeps entries $O(1)$ (variance $2/\pi$). Used for DQN/Dueling.

**Initialisation.** Factorised: $\mu_{i,j}\sim\mathcal{U}[-1/\sqrt{p},\,1/\sqrt{p}]$,
$\sigma_{i,j}=\sigma_0/\sqrt{p}$ with $\sigma_0=0.5$. Independent: $\mu_{i,j}\sim\mathcal{U}[-\sqrt{3/p},\sqrt{3/p}]$,
$\sigma_{i,j}=0.017$.

**RL integration.**
- *DQN / Dueling*: delete $\epsilon$-greedy; act greedily under the randomised network. Replace FC
  head layers with factorised noisy layers. Loss
  $\bar L(\zeta)=\mathbb{E}\,\mathbb{E}_{D}[r+\gamma\max_bQ(y,b,\varepsilon';\zeta^-)-Q(x,a,\varepsilon;\zeta)]^2$,
  with *independent* noise for online ($\varepsilon$), target ($\varepsilon'$), and acting
  ($\varepsilon''$) to avoid TD-target bias; noise resampled each step, fixed across a replay batch.
  Dueling uses the double-DQN target $b^\star(y)=\arg\max_bQ(y,b,\varepsilon'';\zeta)$.
- *A3C*: remove the entropy bonus; make policy/value head layers noisy (independent Gaussian).
  Fix the noise for the whole $n$-step rollout ($\varepsilon_i\equiv\varepsilon$) to keep the
  on-policy return estimate consistent; resample after each optimisation step.

**Cost.** Doubles linear-layer parameters; forward cost still dominated by the weight×activation
matmul, so overhead is marginal.

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):  # std_init = sigma_0
        super().__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters(); self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())               # f(x) = sgn(x) sqrt(|x|)

    def reset_noise(self):
        eps_in, eps_out = self._scale_noise(self.in_features), self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))      # outer product (factorised, rank-one)
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu              # deterministic at eval
        return F.linear(x, w, b)

class NoisyDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.fc   = NoisyLinear(3136, 512)
        self.head = NoisyLinear(512, n_actions)
    def forward(self, x):
        return self.head(F.relu(self.fc(self.torso(x / 255.0))))
    def reset_noise(self):
        self.fc.reset_noise(); self.head.reset_noise()

def act(net, x):                       # no epsilon-greedy; greedy under perturbed net
    net.reset_noise()
    with torch.no_grad():
        return net(x).argmax(1)

def loss_fn(online, target, obs, actions, rewards, next_obs, dones, gamma):
    online.reset_noise(); target.reset_noise()              # independent online/target noise
    q = online(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        y = rewards + gamma * target(next_obs).max(1).values * (1.0 - dones)
    return F.mse_loss(q, y)
```
