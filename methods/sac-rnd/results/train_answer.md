Offline reinforcement learning has one disease that organizes everything else, and any method here has to be built directly against it. I have a fixed dataset $D$ of transitions collected by some unknown behavior policy, no environment to query, and I want a policy better than the one that produced the data. The standard machinery is a bootstrapped critic: regress $Q(s,a)$ toward $r + \gamma\,Q(s',a')$ where $a'$ is whatever the policy proposes at the next state. Online this self-corrects — if the critic over-values an action, the agent eventually tries it and sees the truth. Offline there is no "eventually." The actor proposes actions that were never in $D$ at $s'$, the critic has no data there and extrapolates, and the extrapolation is almost always *upward*, because the actor is optimized to climb wherever the critic bulges and the bulges are disproportionately the positive-error regions. The inflated value backs up through the Bellman recursion, the policy chases it, and nothing ever pulls it back. So every offline method is at heart a way of keeping the value, and the policy, honest on actions the data does not support.

The cleanest way to *state* what I want is in the language of uncertainty. If I had, for every $(s,a)$, a reliable number $b(s,a)$ measuring how unfamiliar that action is at that state — small on the data, large off it — I could simply subtract a multiple of it from the value and from the policy objective: be pessimistic exactly where I am ignorant. This is the anti-exploration framing, and it tells me where to look. Online exploration *adds* a novelty bonus $+b$ to chase the unknown; offline I want to *subtract* the same novelty signal $-b$ to flee it. The two are one estimator with opposite sign, so the question reduces to: what is a good, cheap novelty estimator over $(s,a)$ pairs? The ensemble answer is known and strong — keep $N$ critics, initialize them differently, and read their disagreement at $(s,a)$ as the uncertainty; SAC-N and EDAC do exactly this and sit near the top of D4RL. But there the uncertainty signal *is* the ensemble, so the cost scales with $N$, ten or fifty critics, and when the contribution must be algorithmic rather than a matter of capacity, an $N$-critic ensemble is precisely the move I cannot make. I want ensemble-quality pessimism from a single small extra network. The other ensemble-free families do not give me a per-$(s,a)$ uncertainty either: policy-constraint methods (TD3+BC, ReBRAC) clone the behavior policy with a *fixed* notion of "near the data" and pull toward dataset actions uniformly; value-regularization (CQL) must explicitly sample and push down OOD actions through a tuned temperature; expectile methods (IQL) stay in-sample but are conservative and leave value on the table on tasks that need aggressive exploitation.

I propose SAC-RND: anti-exploration by Random Network Distillation, built on a Soft Actor-Critic base. The novelty signal comes from RND, the canonical ensemble-free detector. Keep a fixed, randomly-initialized *target* network $g$ and train a *predictor* $\hat g$ by regression to match $g$ on the inputs actually observed; on inputs seen often the predictor learns to mimic the target and the squared error $\lVert \hat g - g\rVert^2$ goes to zero, while on inputs never seen it was never trained and the error stays large. One frozen network, one trained network, a squared error, and a novelty score. The bet is that computing this error over $(s,a)$ inputs, training the predictor only on dataset $(s,a)$, yields a clean OOD-action detector: small on in-data actions, large on the actions the offline policy must be kept away from. SAC is the natural continuous base because its Tanh-Gaussian stochastic actor already samples the actions I need to score, and it brings twin critics with a $\min$ target, LayerNorm, and an automatically tuned entropy temperature.

The defining detail — the thing that turns this from a known-to-fail idea into a working one — is the conditioning, not the loss. A naive prior that concatenates the action onto the state at the input, $[s,a]\to\hat g$, is reported to be "not discriminative enough," and the reason is worth tracing because it dictates the architecture. The predictor is a flexible MLP trained to match a fixed target on the dataset, while the actor is *simultaneously* being optimized to make $b(s,\pi(s))$ small, since I subtract $\beta\, b$ from its objective, and the actor controls the action argument. If the action enters merely as a concatenated input, the error surface as a function of $a$ is smooth and largely featureless away from the data — nothing forces it *high* on a specific OOD action, only low on the data it was trained on. So the actor can find directions in action space where the predictor's error happens to already be small, slide the bonus down, and never return to the data. The penalty is escapable. The repair is to make the predictor's error a *sharp* function of the action, genuinely small on dataset actions and large the instant the action leaves the data, so that the only way to drive $b$ down is to propose an in-distribution action. The lever for that is feature-wise linear modulation. Let one input be the feature stream flowing through the MLP and let the other input produce, through its own small linear map, per-unit scale and shift parameters $(\gamma,\beta)$ that multiply and offset a hidden layer,
$$h \leftarrow \gamma \odot h + \beta.$$
Now the conditioning input does not merely add a few dimensions at the front; it *reshapes* the function the network computes. A different action yields a different $(\gamma,\beta)$ and hence a different target embedding to match, so the predictor's error becomes genuinely action-dependent — it can be small on the $(s,a)$ pairs it was trained on and is *forced* to be different, hence untrained and large, on action settings the data never produced. The multiplicative gating is exactly what denies the actor a smooth low-error escape route. In the implementation the action is the FiLM feature and the state is the context, but the roles can be swapped with the same effect; what matters is the multiplicative conditioning. This is an *architecture* fix, and the ablation that proves it is the direct comparison of concatenation against FiLM (and gated/bilinear variants, at the first versus last hidden layer), watching the OOD-detection quality move.

With a discriminative $b$ in hand the placement matters as much as the signal, because OOD actions corrupt the value by two distinct routes and I want to close both. The first is the actor: at state $s$ the policy proposes $\pi(s)$, possibly OOD, and the actor objective rewards climbing $Q(s,\pi(s))$, so I subtract $\beta\, b(s,\pi(s))$ from the actor loss — the policy is paid to climb $Q$ but charged for unfamiliarity, pulling it toward actions that are both high-value and in-distribution. The second route is the bootstrap itself: the critic target evaluates $Q(s',a')$ at the next sampled action $a'\sim\pi(\cdot\mid s')$, which may already be OOD irrespective of the actor this step, and that inflated next-value backs up through every Bellman step. So I subtract $\beta\, b$ inside the target as well,
$$y = r + \gamma\,(1-d)\,\big[\textstyle\min_i Q_i^{\text{tgt}}(s',a') - \alpha\,\log\pi(a'\mid s') - \beta\, b(s',a')\big],$$
which suppresses the over-valuation at its source. Penalizing at both the actor and the target is the exact anti-exploration mirror of how an online bonus would be added to both the policy reward and the value.

A few details are easy to get wrong and are load-bearing. The bonus scale drifts over training — early on the predictor matches nothing and $b$ is enormous, later it shrinks — so a fixed $\beta$ would mean wildly different effective conservatism over a run; I divide $b$ by a running standard deviation of the raw distillation error so that $\beta$ controls a stable, unit-free penalty, $b(s,a) = \lVert \hat g(s,a) - g(s,a)\rVert^2 / \mathrm{std}$. The target network must stay frozen, with no gradient — it is the fixed random reference; only the predictor trains, and only on dataset $(s,a)$, never on the actor's proposed actions, or the system could game itself by making the reference easy. The SAC base keeps its entropy temperature $\alpha$ tuned to a target entropy of $-\dim(\mathcal A)$, the twin critics keep LayerNorm and the $\min$, and targets are Polyak-updated with $\tau = 5\times 10^{-3}$. Finally $\beta$ is the one knob that genuinely needs per-dataset setting, because how aggressively to flee OOD actions depends on how much the dataset covers: near-expert data wants a light penalty (stay close, exploit), a broad mediocre dataset can tolerate or even needs a heavier one — so $\beta$ is swept per dataset (e.g. halfcheetah-medium $0.3$, walker2d-medium $8.0$) rather than fixed. The result is ensemble-quality pessimism — matching SAC-N and EDAC on D4RL Gym and clearing the ensemble-free TD3+BC, IQL, and CQL — bought with one small extra network instead of $N$ critics.

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FiLMMLP(nn.Module):
    """MLP over `feature`, FiLM-modulated by `context`."""
    def __init__(self, feature_dim, context_dim, out_dim, hidden=256):
        super().__init__()
        self.film = nn.Linear(context_dim, 2 * hidden)
        self.l1 = nn.Linear(feature_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, feature, context):
        gamma, beta = torch.chunk(self.film(context), 2, dim=-1)
        h = F.relu(gamma * self.l1(feature) + beta)
        h = F.relu(self.l2(h))
        return self.out(h)


class RND(nn.Module):
    """Anti-exploration novelty over (s, a): action conditions the prior."""
    def __init__(self, obs_dim, act_dim, embedding_dim=32):
        super().__init__()
        self.predictor = FiLMMLP(act_dim, obs_dim, embedding_dim)
        self.target = FiLMMLP(act_dim, obs_dim, embedding_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        self.register_buffer("rms_var", torch.ones(()))

    def _embed(self, s, a):
        pred = self.predictor(a, s)
        with torch.no_grad():
            targ = self.target(a, s)
        return pred, targ

    def bonus(self, s, a):
        pred, targ = self._embed(s, a)
        return ((pred - targ) ** 2).sum(-1) / (torch.sqrt(self.rms_var) + 1e-8)

    def distill_loss(self, s, a):
        pred, targ = self._embed(s, a)
        raw = ((pred - targ) ** 2).sum(-1)
        with torch.no_grad():
            self.rms_var.mul_(0.99).add_(0.01 * raw.var(unbiased=False))
        return raw.mean()


class TanhGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_sigma = nn.Linear(hidden, act_dim)

    def sample(self, s):
        h = self.trunk(s)
        dist = Normal(self.mu(h), self.log_sigma(h).clamp(-5.0, 2.0).exp())
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(-1) - torch.log(1 - torch.tanh(raw) ** 2 + 1e-6).sum(-1)
        return torch.tanh(raw), logp

    @torch.no_grad()
    def act(self, s):
        return torch.tanh(self.mu(self.trunk(s)))


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 1))
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


def update(batch, actor, critics, critic_targs, rnd, log_alpha, target_entropy,
           opts, beta, gamma=0.99, tau=5e-3):
    s, a, r, s2, done = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
    alpha = log_alpha.exp()

    rnd_loss = rnd.distill_loss(s, a)
    opts["rnd"].zero_grad(); rnd_loss.backward(); opts["rnd"].step()

    with torch.no_grad():
        a2, logp2 = actor.sample(s2)
        q2 = torch.min(critic_targs[0](s2, a2), critic_targs[1](s2, a2))
        q2 = q2 - alpha * logp2 - beta * rnd.bonus(s2, a2)
        target = r + (1 - done) * gamma * q2
    c_loss = sum(F.mse_loss(c(s, a), target) for c in critics)
    opts["critic"].zero_grad(); c_loss.backward(); opts["critic"].step()

    pi, logp = actor.sample(s)
    q_pi = torch.min(critics[0](s, pi), critics[1](s, pi))
    a_loss = (alpha.detach() * logp + beta * rnd.bonus(s, pi) - q_pi).mean()
    opts["actor"].zero_grad(); a_loss.backward(); opts["actor"].step()

    alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
    opts["alpha"].zero_grad(); alpha_loss.backward(); opts["alpha"].step()

    for c, ct in zip(critics, critic_targs):
        for p, tp in zip(c.parameters(), ct.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)
# beta: per-dataset (halfcheetah-medium 0.3, walker2d-medium 8.0); target_entropy = -act_dim
```
