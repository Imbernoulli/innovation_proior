The distributional object was the strongest single change to the floor — 164% median, the best any *one* modification bought. I now have six improvements, each measured in isolation against the DQN floor: decoupling the target ($79\to117$), learned parametric exploration ($117\to118$), prioritized replay ($118\to140$), the dueling head ($140\to151$), and the distributional value object ($151\to164$). The question implicit since rung 2 is whether they compound. If each adds a real and *independent* improvement, an agent with all six should sit well above the best single component's 164%. But "independent" is load-bearing and cannot be assumed — these are not orthogonal knobs in a config file, they are changes to *overlapping* parts of the same agent. Running down what each touches: the decoupled target changes *how the bootstrap action is selected and evaluated*; prioritized replay changes *which transitions I sample*; the dueling head changes *the architecture that produces the values*; learned noise changes *how I explore*; the distributional object changes *what the values are*. Five genuinely different surfaces — target construction, sampling, architecture, exploration, representation — so there is no reason fixing the sampling distribution should undo fixing the target's bias, or a better-shaped head conflict with learned exploration. The improvements are *largely* independent because they address *largely* independent weaknesses, which is the precondition for compounding rather than cancelling.

So I propose **Rainbow**: combine all six in one agent, with the categorical return distribution as the central object — but the combination is a design, not a bolt-on, because three components were specified for a *scalar* and have to be re-expressed over the distribution. *Double Q-learning over a distribution*: the decoupled target needs a greedy next action, but greedy with respect to what, now that there is no scalar $Q$? I keep the policy mean-greedy, so I select the bootstrap action by the *online* net's mean, $a^\star=\arg\max_a z^\top p_\theta(S',a)$, and evaluate by taking the *target* net's whole *distribution* for that action, $p_{\bar\theta}(S',a^\star)$ — the decoupling preserved, now at the level of distributions. *Dueling over a distribution*: the value/advantage split was on scalars, so I lift it to the *logits*, forming each action's $N$ atom-logits as a per-atom value stream plus a per-atom advantage stream with the mean-over-actions subtracted, $\text{logits}_i(x,a)=V_i(x)+A_i(x,a)-\frac1{|\mathcal A|}\sum_{a'}A_i(x,a')$, and *then* softmaxing over the atom dimension to get $p_i(x,a)$ — the identifiability-fixing aggregator on the logits, the categorical structure recovered by the softmax, so the dueling architecture and the distributional head are the same head. *Prioritized replay over a distribution*: there is no scalar TD error anymore; the loss is the per-sample cross-entropy/KL between projected target and prediction, so I make the *KL loss itself* the priority source, $p_i\propto L_t^\omega$ — actually the *more* principled quantity, measuring how surprising the whole return distribution is rather than just its mean, and already computed for the gradient. *Learned noise over everything*: the noisy linear layers replace the fully-connected layers of the now-dueling, now-distributional head, and acting is greedy under the sampled net with $\epsilon=0$ — $\epsilon$-greedy gone entirely, no conflict because the noise is on the head's weights, orthogonal to what the head computes.

Assembling these surfaces one more lever I could not justify as its own rung — it is not in the per-component ablation — yet composes naturally here and is nearly free given the distributional machinery: **multi-step returns**. Every rung so far bootstrapped after *one* step, $r+\gamma(\cdot)$, a target maximally biased toward the current (wrong) value estimate that propagates reward backward only one state per update — slow credit assignment. An $n$-step target $R_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}$ followed by a bootstrap $\gamma^n(\cdot)$ uses $n$ steps of *real* reward before trusting the estimate, trading a little variance for much faster reward propagation and less reliance on the early, badly-wrong value function. Over the distribution this is the cleanest possible edit: shift the target atoms by $R_t^{(n)}$ and contract them by $\gamma_t^{(n)}=\gamma^n$ before projecting, $\hat{\mathcal T}z_j=R_t^{(n)}+\gamma^n z_j$ — exactly the C51 shift-and-scale with $r\to R_t^{(n)}$ and $\gamma\to\gamma^n$, then the same projection $\Phi$ and the same cross-entropy. $n=3$ is the balance: long enough to speed credit assignment, short enough that the off-policy-ness of an $n$-step return over a few stale actions is tolerable.

So the integrated target and loss are
$$d_t^{(n)}=\Big(R_t^{(n)}+\gamma_t^{(n)}z,\;p_{\bar\theta}\big(S_{t+n},\arg\max_a z^\top p_\theta(S_{t+n},a)\big)\Big),\qquad L_t=-\sum_i\big(\Phi_z d_t^{(n)}\big)_i\log p^i_\theta(S_t,A_t):$$
take the online net's mean to pick $a^\star$, take the target net's distribution there, shift it by the $n$-step return and contract by $\gamma^n$, project back onto the fixed $[-10,10]$ $51$-atom grid, and minimize the cross-entropy of the projected target against the prediction — with the per-sample loss feeding both the gradient (importance-weighted, $\beta$-annealed) and the replay priority (priorities $\propto L_t^\omega$; the importance weights multiply the minibatch loss but are *not* folded into the stored priority), on a dueling-over-logits, noisy-layer head. The hyperparameters that make a *single* set of them work across all 57 games are the careful part: Adam at $6.25\times10^{-5}$ (lower, because prioritization and the multi-step target both raise effective gradient magnitude), Adam $\epsilon=1.5\times10^{-4}$, target sync every 32K frames, priority exponent $\omega=0.5$ with importance-sampling $\beta:0.4\to1.0$, $n=3$, $51$ atoms on $[-10,10]$, noisy-layer $\sigma_0=0.5$, and $\epsilon=0$ for acting. The re-expressions above — distributional priority, logit-level dueling, mean-greedy distributional double selection, the shared learning-rate cut — are precisely the integration points where a naive concatenation of six papers would collide, and what keeps them composing instead.

The bar, stated so I can be proven wrong: the floor was 79%, the best single component the distributional object at 164%. If the six were perfectly redundant — all attacking the same underlying weakness — the combined agent would do no better than that 164%, the null this rung tests against. If instead they are largely independent, it should clear 164% *decisively* and roughly double the floor into the low 200s, the level no single-component agent came close to — six largely independent fixes to six largely independent weaknesses of the DQN floor, re-expressed so they share one distributional, dueling, noisy, prioritized, multi-step agent: the best value-based learner on the suite.

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
