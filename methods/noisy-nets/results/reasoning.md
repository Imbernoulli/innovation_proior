OK, let me think about exploration from scratch, because I keep being bothered by how thin the standard story is. In a value-based agent I have a network $Q(x,a;\theta)$, I act greedily, and to explore I flip a coin with probability $\epsilon$ and take a uniformly random action. In an actor-critic I sample from a softmax policy and add an entropy bonus to keep it from sharpening too fast. Both of these are the same move dressed differently: take the agent's decision and corrupt it with noise that is freshly drawn at every single time step and, for $\epsilon$-greedy, doesn't even look at the state. Stare at what that actually does over a trajectory. At step $t$ I might jerk left by accident; at step $t{+}1$ the coin comes up greedy again and I go back to whatever the policy wanted; at $t{+}2$ another independent jerk. The perturbations are temporally decorrelated, so they cancel out into a kind of Brownian wiggle around the greedy trajectory. To actually reach a far-away part of the state space I'd need a *coherent* sequence of off-policy actions — go left, and keep going left, for twenty steps, because that's the only way to discover the room behind the door. Independent per-step coins essentially never produce a coherent twenty-step deviation; the probability of twenty "explore" coins all pointing the same useful direction is vanishing. So dithering can polish behaviour locally but can't commit to an exploratory *plan*.

And there's a second thing nagging me: the amount of exploration is a number I set by hand. The $\epsilon$ schedule, the entropy weight $\beta$ — I pick them, hold them fixed across wildly different games, and anneal $\epsilon$ on a clock that has nothing to do with whether *this* agent in *this* game is still uncertain. That's clearly wrong in spirit. A well-explored region should get little perturbation; a novel one should get a lot; and the agent itself is the only thing that knows which is which.

So what would fix both at once? I want perturbations that (1) are consistent over a stretch of time so they add up to a real strategy, (2) depend on the state so the deviation is structured rather than uniform noise, and (3) have a magnitude the agent *learns* rather than one I dial in. Let me think about where to inject the noise so I get all three for free.

The dithering schemes perturb the *output* — the action. That's exactly why they're decorrelated and state-blind: each action draw is its own independent event. What if I perturb the *parameters* instead? Suppose I hold a single perturbation of the weight vector fixed for a whole episode and act greedily under the perturbed network. Now property (1) falls out immediately: the perturbation is the same at every step, so the induced behaviour is a single coherent policy for the whole episode, not a fresh dice roll each step. Property (2) falls out too, and this is the part I find pretty: a fixed change $\delta\theta$ to the weights does *not* produce a fixed change to the action — it produces a change that flows through the network and depends on the input $x$. The same $\delta\theta$ nudges the chosen action one way in state $A$ and a different way in state $B$, because the Jacobian $\partial Q/\partial\theta$ is state-dependent. So a single weight perturbation is automatically a *state-dependent, temporally consistent* exploratory policy. That's "deep exploration" — commit to a perturbed value function, act greedily and consistently under it — and it's the randomised-value-function idea, except I want to get it without duplicating chunks of the network the way bootstrapped value heads do.

Now property (3): I want the *scale* of the perturbation to be learned. So don't just add fixed noise. Write each weight as a mean plus a learnable scale times a noise draw. Let me set this up carefully. For a parameter vector $\theta$ I write
$$\theta \;=\; \mu + \Sigma\odot\varepsilon,$$
where $\mu$ is the usual learnable weight, $\Sigma$ is a vector of *learnable* per-parameter noise scales (same shape as $\mu$), $\varepsilon$ is a zero-mean fixed-statistics noise of the same shape, and $\odot$ is elementwise. Call the genuine learnables $\zeta=(\mu,\Sigma)$. The agent's loss $L(\theta)$ is now random because $\theta$ is random; the thing I actually want to descend is its expectation over the noise,
$$\bar L(\zeta) \;=\; \mathbb{E}_\varepsilon\big[L(\mu+\Sigma\odot\varepsilon)\big].$$
Can I get a gradient of this in $\mu$ and $\Sigma$? Because $\varepsilon$'s statistics don't depend on $\zeta$ (I drew it from a fixed distribution and then formed $\theta$ as an affine function of it), the expectation and the gradient commute:
$$\nabla_\zeta\bar L(\zeta) \;=\; \mathbb{E}_\varepsilon\big[\nabla_{\mu,\Sigma}\,L(\mu+\Sigma\odot\varepsilon)\big],$$
and I can estimate it with a single Monte-Carlo sample $\xi$ per optimisation step,
$$\nabla_\zeta\bar L(\zeta)\;\approx\;\nabla_{\mu,\Sigma}\,L(\mu+\Sigma\odot\xi).$$
This is the reparameterised gradient: the noise sits *outside* the parameters being differentiated, so ordinary backprop flows through $\mu$ and $\Sigma$. The gradient on $\Sigma$ is the crucial one — it tells the RL loss itself how much noise it wants. If reducing the noise on a given weight lowers the expected TD error, $\Sigma$ on that weight shrinks; if the loss is insensitive (or exploration is paying off) it can stay large. The exploration intensity is now a parameter trained by the same signal as everything else. That kills the hand-tuned-$\epsilon$ problem, and it does it per-weight and per-state-implicitly, which is exactly the contextual exploration I wanted.

Let me write the noisy layer concretely, because the matmul structure is going to matter for cost. A standard linear layer with $p$ inputs and $q$ outputs is $y=wx+b$, $w\in\mathbb{R}^{q\times p}$, $b\in\mathbb{R}^q$. Replace $w$ and $b$ by their noisy versions:
$$y \;=\; (\mu^w+\sigma^w\odot\varepsilon^w)\,x \;+\; \mu^b+\sigma^b\odot\varepsilon^b,$$
with learnables $\mu^w,\sigma^w\in\mathbb{R}^{q\times p}$ and $\mu^b,\sigma^b\in\mathbb{R}^{q}$, and noise $\varepsilon^w\in\mathbb{R}^{q\times p}$, $\varepsilon^b\in\mathbb{R}^q$. The simplest choice is to make every entry of $\varepsilon^w$ and $\varepsilon^b$ an independent unit Gaussian. That needs $pq+q$ fresh Gaussian draws per layer per step. For a distributed actor-critic where many actors run in parallel and compute is cheap, fine. But for a single-thread agent like DQN that takes an optimisation step every action, drawing a fresh $pq$-entry Gaussian matrix every step is a real cost — the random-number generation alone competes with the matmul. I'd like the same kind of weight noise for far fewer draws.

The trick is to *factorise* the weight noise. Instead of $pq$ independent entries, generate one noise per input, $\varepsilon_i$, $i=1\dots p$, and one per output, $\varepsilon_j$, $j=1\dots q$ — $p+q$ draws total — and build the matrix entry as an outer product:
$$\varepsilon^w_{i,j} \;=\; f(\varepsilon_i)\,f(\varepsilon_j),\qquad \varepsilon^b_j \;=\; f(\varepsilon_j),$$
for some scalar function $f$. Now the per-step RNG cost drops from $pq$ to $p+q$, which for a 512-wide layer is the difference between a quarter-million draws and a thousand. The weight-noise matrix is rank-one in the noise (an outer product), which is a real reduction in the *richness* of the noise, but the empirical bet is that rank-one weight perturbation is plenty for exploration, and the saving is what makes it usable in a single-thread loop. I reuse the output factor $f(\varepsilon_j)$ for the bias rather than giving the bias its own noise — I could have used $f(x)=x$ just for the bias, but keeping the same output-noise factor for weights and bias is simpler and ties the bias perturbation to the same "this output unit is being perturbed" event.

What should $f$ be? If I naively set $f(x)=x$, then $\varepsilon^w_{i,j}=\varepsilon_i\varepsilon_j$ is a product of two independent unit Gaussians. That product has mean $0$ but it's heavy-tailed — its variance is $1$ but it has excess kurtosis, and occasional entries are large because both factors happened to be large. I'd rather each *factor* contribute something of unit-ish, well-controlled scale so the outer product doesn't get spiky. Take $f(x)=\operatorname{sgn}(x)\sqrt{|x|}$. Then $f(\varepsilon)$ has the sign of $\varepsilon$ and magnitude $\sqrt{|\varepsilon|}$, and its second moment is $\mathbb{E}[f(\varepsilon)^2]=\mathbb{E}[|\varepsilon|]=\sqrt{2/\pi}\approx0.80$ for a unit Gaussian. So each weight-noise entry has $\mathbb{E}[\varepsilon^w_{i,j}]=\mathbb{E}f(\varepsilon_i)\,\mathbb{E}f(\varepsilon_j)=0$ (zero-mean, good — I don't want to bias the weights) and variance $\mathbb{E}[f(\varepsilon_i)^2]\mathbb{E}[f(\varepsilon_j)^2]=2/\pi\approx0.64$, a tame $O(1)$ number, with the square-root compressing the tails of each factor so the product doesn't blow up. That's why the square-root-of-magnitude shows up; it's a variance-control choice on the factorisation, not anything deep.

Now initialisation, and here the two noise schemes need different constants, which took me a second to see why. The mean weights $\mu^w$ I want initialised the usual way so that at the start the *signal* through the layer has unit-ish scale: a fan-in initialisation with variance $\propto1/p$. For the independent scheme I draw $\mu_{i,j}\sim\mathcal{U}[-\sqrt{3/p},\sqrt{3/p}]$ — a uniform with variance $\frac{1}{3}\cdot\frac{3}{p}=1/p$, the standard fan-in scale. For the factorised scheme I draw $\mu_{i,j}\sim\mathcal{U}[-1/\sqrt{p},1/\sqrt{p}]$ instead — variance $\frac{1}{3p}$, three times smaller. The reason is that the factorised noise carries a larger *effective* perturbation per weight (the rank-one structure correlates the entries within a row/column), so I pull the initial means in to keep the perturbed weights from starting too hot. For the noise scale $\sigma$: in the factorised case set every $\sigma_{i,j}=\sigma_0/\sqrt{p}$ with $\sigma_0=0.5$, so the initial injected noise on a weight has standard deviation that, after the $1/\sqrt{p}$ fan-in scaling, sits at a sensible fraction of the signal — the $1/\sqrt p$ makes the *total* perturbation reaching an output unit (summed over $p$ inputs) independent of layer width. In the independent (unfactorised) case I instead just set $\sigma_{i,j}=0.017$, a flat constant carried over from the supervised noisy-network setting; it wasn't tuned, the claim is only that any value on that scale behaves similarly. The asymmetry $\sqrt{3/p}$ vs $1/\sqrt p$ in the means, and $\sigma_0/\sqrt p$ vs flat $0.017$ in the scales, is entirely about matching the effective noise magnitude of the two schemes at initialisation.

Now wire this into the agents. For DQN and dueling I do two things. First — and this is the satisfying part — I *delete $\epsilon$-greedy entirely*. There's no separate exploration rule anymore; the agent acts greedily with respect to its own *randomised* value function. The randomness in the weights is the only source of exploration. Second, I replace the fully-connected layers in the heads with noisy layers (factorised noise, because DQN is single-thread). The noise sample is held fixed across a replay batch, and since DQN takes one optimisation step per environment step, I resample the weight noise before each action — so within an episode the agent is following one coherent perturbed policy until the next learning step nudges it.

The loss needs care about *which* noise goes where. Writing the noisy action-value as a random function $Q(x,a,\varepsilon;\zeta)$, the NoisyNet-DQN loss is
$$\bar L(\zeta)=\mathbb{E}\Big[\;\mathbb{E}_{(x,a,r,y)\sim D}\big[\,r+\gamma\max_b Q(y,b,\varepsilon';\zeta^-)-Q(x,a,\varepsilon;\zeta)\,\big]^2\Big],$$
and the thing to get right is that the online network uses one noise sample $\varepsilon$ and the target network uses an *independent* sample $\varepsilon'$. If I shared the noise between online and target, the two $Q$ values would be correlated through the same $\varepsilon$ and the TD target would inherit a bias; drawing them independently keeps the bootstrap target an unbiased-in-the-noise estimate. And the action I actually take when collecting data uses yet another independent sample $\varepsilon''$ — I act greedily under a freshly perturbed online network. For dueling the only change is the target follows the double-DQN form: the online net (under $\varepsilon''$) selects $b^\star(y)=\arg\max_b Q(y,b,\varepsilon'';\zeta)$ and the target net (under $\varepsilon'$) evaluates it.

For A3C the integration is analogous but with one on-policy subtlety I have to respect. I remove the entropy bonus — its whole job was to keep the policy stochastic for exploration, and now the weight noise does that, more coherently. I make the head's linear layers noisy (independent Gaussian here, since A3C is distributed and RNG cost isn't the bottleneck). But A3C is *on-policy*: it forms $n$-step return estimates $\hat Q_i=\sum_{j=i}^{k-1}\gamma^{j-i}r_{t+j}+\gamma^{k-i}V(x_{t+k};\zeta_V,\varepsilon_i)$ and computes a policy gradient under the policy that generated the rollout. If I let the noise vary *within* a rollout — a different $\varepsilon_i$ at each step — then the value bootstrap and the policy that generated the actions wouldn't correspond to a single fixed policy, and the return estimate would be inconsistent. So I fix the noise for the *entire rollout*: $\varepsilon_i=\varepsilon$ for all $i$. One perturbed policy generates the whole $k$-step rollout, the gradient is taken under that same perturbed policy, and the noise is resampled only after the optimisation step (which for A3C happens every $n$ steps). That preserves on-policy consistency while still getting episode-coherent exploration.

Let me sanity-check the worry that this just collapses back to a deterministic net. Since the loss $L(\zeta)$ is a positive continuous function of $\zeta$, a *deterministic* optimiser exists — the network could in principle drive every $\sigma$ to zero and recover an ordinary net. So would gradient descent simply learn to switch off the noise? Not necessarily, and that's the point: the $\sigma$ gradient is driven by the *task*, so the noise persists exactly where keeping it lowers the long-run loss (i.e. where exploration is still paying) and decays where the agent is confident. If I were to track $\bar\Sigma=\frac1N\sum_i|\sigma^w_i|$ per layer over training, I'd expect it to *anneal itself* — and crucially to do so differently across games and even across seeds, because the schedule is now emergent from the learning dynamics rather than a clock I set. That self-tuning, per-state, per-weight annealing of exploration is the whole payoff, and it costs me only a doubling of the linear-layer parameters (a $\mu$ and a $\sigma$ each) — cheap, because the weight is an affine function of noise and the forward cost is still dominated by the weight-times-activation matmul, not by forming the weights.

Now the code. The heart is a `NoisyLinear` that owns $\mu$, $\sigma$, and a buffer for the current noise sample, can resample the factorised noise, and at eval time falls back to the means (act with $\mu$, no exploration). Everything else is the ordinary DQN loop with `nn.Linear` swapped for `NoisyLinear` and $\epsilon$-greedy deleted.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    # y = (mu_w + sigma_w ⊙ eps_w) x + (mu_b + sigma_b ⊙ eps_b)
    # mu_*, sigma_* are learnable; eps_* are resampled fixed-statistics noise.
    def __init__(self, in_features, out_features, std_init=0.5):  # std_init = sigma_0
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.std_init = std_init
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # factorised init: mu ~ U[-1/sqrt(p), 1/sqrt(p)],  sigma = sigma_0 / sqrt(p)
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())          # f(x) = sgn(x) sqrt(|x|)

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)  # p draws
        eps_out = self._scale_noise(self.out_features) # q draws
        self.weight_epsilon.copy_(eps_out.ger(eps_in)) # eps_w[i,j] = f(eps_i) f(eps_j): rank-one
        self.bias_epsilon.copy_(eps_out)               # eps_b[j]   = f(eps_j)

    def forward(self, x):
        if self.training:                              # explore: perturbed weights
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:                                          # eval: means only, deterministic
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

class NoisyDQN(nn.Module):
    # DQN torso unchanged; FC head built from NoisyLinear -> noise is the only exploration.
    def __init__(self, n_actions):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc   = NoisyLinear(3136, 512)
        self.head = NoisyLinear(512, n_actions)

    def forward(self, x):
        h = F.relu(self.fc(self.torso(x / 255.0)))
        return self.head(h)                            # Q(x,·, eps; zeta)

    def reset_noise(self):
        self.fc.reset_noise(); self.head.reset_noise()

def act(net, x):
    # NO epsilon-greedy: greedy under a freshly perturbed net (eps'' resampled before acting)
    net.reset_noise()
    with torch.no_grad():
        return net(x).argmax(1)

def loss_fn(online, target, obs, actions, rewards, next_obs, dones, gamma):
    online.reset_noise()                               # eps  (online)
    target.reset_noise()                               # eps' (target) -- INDEPENDENT of eps
    q = online(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        # NoisyNet-DQN: max over target net under its own noise eps'
        target_max = target(next_obs).max(1).values
        y = rewards + gamma * target_max * (1.0 - dones)
    return F.mse_loss(q, y)
    # NoisyNet-Dueling variant: online selects b* = argmax_b Q(y,b,eps''; zeta),
    # target evaluates Q(y, b*, eps'; zeta^-)  (double-DQN target).
```

The causal chain in one breath: dithering ($\epsilon$-greedy / entropy) perturbs the *action* with per-step, state-independent noise, so it can't commit to a multi-step exploratory plan and its scale is a hand-set hyperparameter → perturb the *weights* instead, and a single weight perturbation held fixed over an episode is automatically a coherent, state-dependent exploratory policy → write each weight as $\mu+\sigma\odot\varepsilon$ so the *scale* $\sigma$ is learnable and the reparameterised gradient lets the RL loss tune it per-weight → factorise the weight noise as an outer product $f(\varepsilon_i)f(\varepsilon_j)$ with $f=\operatorname{sgn}\cdot\sqrt{|\cdot|}$ to cut RNG cost from $pq$ to $p+q$ and keep entries $O(1)$ → delete $\epsilon$-greedy and act greedily under the randomised net, using independent noise for online/target/action to avoid TD bias (fixing noise per rollout for on-policy A3C) → the network self-anneals its own exploration, per state and per weight, at no real compute cost.
