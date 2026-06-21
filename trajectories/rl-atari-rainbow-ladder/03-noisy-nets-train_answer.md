Decoupling the target carried the median from 79% to 117% — the typical game is now above the human reference. With the worst arithmetic defect of the floor fixed, the most obviously crude thing left in the agent is exploration. $\epsilon$-greedy ignores the value function with probability $\epsilon$ and samples a uniform action, otherwise acts greedily — a coin flip at the *action output*, re-flipped every step, uncorrelated with the state and with the previous flip. On a game where reward only appears after a specific *sequence* of several deliberate actions — go to a key, then to a door, then through it, with nothing in between — that fails badly: to traverse a $k$-step exploratory detour by chance the coin has to land right roughly $k$ times in a row, a probability that decays like $\epsilon^k$, and on the hard-exploration games $k$ is large enough that it essentially never happens. The randomness is local in two senses, local in time (each step independent, so no coherent multi-step plan forms) and uniform across states (the same $\epsilon$ in a state I understand perfectly and one I have never resolved). And $\epsilon$ is a number I set by hand on a schedule, outside the learning problem, so the agent cannot *learn* that it should still explore in one region and stop in another.

Both complaints point the same direction: move the stochasticity *upstream*, out of the action output and into the function that produces the values. I propose **Noisy Nets** — learned parametric exploration noise. If I perturb the network's *parameters* rather than its chosen action, a single fixed perturbation induces a *different value function*, and because the perturbation flows through the conv encoder and the head its effect on behavior depends on the input state. Hold that perturbation fixed for a stretch and the agent acts according to one coherent, perturbed value function, which can prefer a *consistent* off-greedy action in a given state across the whole stretch rather than a fresh independent dice roll each step. That is exactly the structured, temporally-extended, state-dependent exploration $\epsilon$-greedy cannot produce: a perturbed net can decide "in *this* kind of state, try the door" and stick to it for as long as the sample is held, which is how a multi-step detour actually gets traversed.

What makes the perturbation answer the second complaint is making it *trainable*. A noisy parameter is $\theta=\mu+\sigma\odot\epsilon$, with $\epsilon$ a vector of zero-mean fixed-statistics noise (drawn each time, not learned), $\mu,\sigma$ learnable, $\odot$ elementwise. This is not a posterior — it is a parameterized source of noise whose *scale* $\sigma$ is trained by gradient descent. The objective becomes the expectation over the noise, $\bar L(\mu,\sigma)=\mathbb{E}_\epsilon[L(\mu+\sigma\odot\epsilon)]$, and because the noise distribution does not depend on $\mu,\sigma$ I can pull the gradient inside and estimate it with a single draw (reparameterization): the gradient w.r.t. $\mu$ is the ordinary weight gradient, and the gradient w.r.t. $\sigma$ is that same local gradient multiplied by the sampled noise $\epsilon$. So backprop directly learns, per parameter, whether more or less injected variation lowers the loss — where the perturbation still helps $\sigma$ stays up, and where behavior has settled and noise only hurts the TD loss $\sigma$ is driven toward zero, automatically and per parameter, with no external schedule.

Concretely I replace the fully-connected layers of the head with *noisy linear* layers. For a layer with $p$ inputs and $q$ outputs the map becomes $y=(\mu^w+\sigma^w\odot\epsilon^w)x+(\mu^b+\sigma^b\odot\epsilon^b)$, with $\mu^w,\sigma^w$ of shape $q\times p$ and $\mu^b,\sigma^b$ of shape $q$. Drawing a full $q\times p$ noise matrix per layer every step is the obvious thing but too expensive relative to the matmul on a single-threaded value agent, so I *factor* the noise: draw $p$ input noises and $q$ output noises, pass each through $f(x)=\operatorname{sign}(x)\sqrt{|x|}$, and set $\epsilon^w_{j,i}=f(\epsilon^{\text{out}}_j)\,f(\epsilon^{\text{in}}_i)$ and $\epsilon^b_j=f(\epsilon^{\text{out}}_j)$. The weight-noise tensor is an outer product, so the count of Gaussian draws drops from $pq+q$ to $p+q$; the transform keeps each factor zero-mean and order-one (for $Z\sim N(0,1)$, $\mathbb{E}[f(Z)]=0$ and $\mathbb{E}[f(Z)^2]=\mathbb{E}|Z|=\sqrt{2/\pi}$, so a factorized weight entry has variance $2/\pi$ — order one, not exactly one).

Wiring it into the value learning, I delete the $\epsilon$-greedy schedule entirely and act *greedily* under the current sampled value network — the exploration now comes from the parameter noise. The discipline that matters is when to resample: the rule is to hold one sample fixed between optimization steps, and since this value agent updates once per action, in practice it resamples the noise before each action and holds it across a replay batch. The online net and target net get *independent* noise draws ($\epsilon$ vs $\epsilon'$): sharing a draw between them would correlate the bootstrapped target with the current estimate, exactly the coupling the floor's target network exists to avoid. So the Double-DQN target from the previous rung is kept, now over sampled nets — the online sampled net selects the next action, the target sampled net (independent noise) evaluates it.

I am sober about what this rung will and will not move, because it sets the bar. The benefit is concentrated on the *hard-exploration* games with long reward-free corridors that $\epsilon$-greedy could never traverse, and there it can be large — but those are a *minority* of the 57. On the bulk of the suite, where Double DQN already explores adequately and the bottleneck is value estimation rather than discovery, replacing dithering with learned parameter noise changes little. The median is a robustness statistic, so a method that lifts a minority of games dramatically while leaving the majority where they were moves the *median* only slightly — the median game is not a hard-exploration game. So I expect this to clear 117% but only barely, with the real action hidden in the tails; and a near-flat median is itself informative — it would say exploration was not what capped the *typical* game, sending me next to the axis that touches every game's data efficiency: how the replay buffer samples.

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
