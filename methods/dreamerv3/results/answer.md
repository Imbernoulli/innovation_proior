# DreamerV3

## Problem

Train a single reinforcement-learning agent that masters a wide range of domains —
visual and proprioceptive continuous control, discrete-action arcade games,
procedurally generated 2D and 3D worlds, and a sparse-reward open-world game — with
**one fixed set of hyperparameters**, no per-task reconfiguration. The agent sees an
observation $x_t$ (a $64\times64\times3$ image, a sensor vector, or both) and a scalar
reward $r_t$, emits an action $a_t$ (continuous or discrete), and maximizes the expected
discounted return $\mathbb{E}\!\left[\sum_{\tau\ge0}\gamma^\tau r_{t+\tau}\right]$.

The binding difficulty is not raw performance on any one benchmark but **robustness
across domains**: rewards span dense scores in the hundreds to sparse binary milestones;
returns can be unimodal or multimodal; observations range from pixels where one pixel
matters to 3D scenes drowning in irrelevant texture. A loss scale, a KL weight, an
entropy coefficient, or a value parameterization correct in one regime is wrong in
another. DreamerV3 (the third Dreamer generation) closes this gap with a layer of
**scale-invariant, self-balancing** objectives stacked on top of a world-model agent.

## Key idea

Learn a latent world model, then learn an actor and critic *purely inside its
imagination*. Every objective is rewritten so its correct setting no longer depends on
the arbitrary scale or shape of rewards, returns, or observations:

- **symlog prediction** — a fixed, sign-preserving, ~identity-near-zero transform for
  deterministic vector-observation targets, so a squared loss neither diverges on large
  targets nor stagnates (as Huber does) nor injects non-stationarity (as running-stat
  normalization does).
- **symexp two-hot regression** — the critic and reward head are *classifiers* over
  exponentially spaced bins trained with soft (two-hot) cross-entropy, decoupling
  gradient size from target size and handling multimodal returns.
- **KL balancing + free bits** — split the latent KL into a dynamics term (prior chases
  posterior, weight 1) and a representation term (posterior yields to prior, weight 0.1),
  each floored at 1 nat, so the representation stays informative without a per-domain
  regularizer and posterior collapse is structurally prevented.
- **percentile return normalization** — divide returns by the 5th–95th percentile range
  (EMA-smoothed), floored at 1, so a single fixed entropy scale explores when rewards are
  sparse and exploits when they are dense, invariant to reward rescaling.
- **1% unimix** — encoder, dynamics-predictor, and actor categoricals are 99% network
  softmax + 1% uniform, preventing KL spikes and infinite log-probs.
- one **Reinforce** actor update covers discrete and continuous actions; an
  **EMA-regularized** distributional critic bootstraps stably; **AGC + LaProp** decouple
  the optimizer from loss scale.

## World model (RSSM, discrete categorical latents)

An encoder maps $x_t$ to a stochastic representation $z_t$ (a vector of categoricals);
a sequence model carries a deterministic recurrent state $h_t$; the model state is
$s_t=\{h_t,z_t\}$, from which reward, continuation, and observation are predicted:

$$
\begin{aligned}
\text{Sequence model:} \quad & h_t = f_\phi(h_{t-1},z_{t-1},a_{t-1}) \\
\text{Encoder:} \quad & z_t \sim q_\phi(z_t \mid h_t, x_t) \\
\text{Dynamics predictor:} \quad & \hat z_t \sim p_\phi(\hat z_t \mid h_t) \\
\text{Reward / continue / decoder:} \quad & \hat r_t,\ \hat c_t,\ \hat x_t \sim p_\phi(\cdot \mid h_t, z_t)
\end{aligned}
$$

Representations are sampled from a vector of softmaxes with straight-through gradients.
The world-model loss, with $\beta_{\mathrm{pred}}=1$, $\beta_{\mathrm{dyn}}=1$,
$\beta_{\mathrm{rep}}=0.1$:

$$
\mathcal{L}(\phi)=\mathbb{E}_{q_\phi}\!\Big[\textstyle\sum_{t}
\beta_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}
+\beta_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}
+\beta_{\mathrm{rep}}\mathcal{L}_{\mathrm{rep}}\Big]
$$

$$
\begin{aligned}
\mathcal{L}_{\mathrm{pred}} &= -\ln p_\phi(x_t\mid z_t,h_t)-\ln p_\phi(r_t\mid z_t,h_t)-\ln p_\phi(c_t\mid z_t,h_t)\\
\mathcal{L}_{\mathrm{dyn}} &= \max\!\big(1,\ \mathrm{KL}[\,\mathrm{sg}(q_\phi(z_t\mid h_t,x_t))\,\|\,p_\phi(z_t\mid h_t)\,]\big)\\
\mathcal{L}_{\mathrm{rep}} &= \max\!\big(1,\ \mathrm{KL}[\,q_\phi(z_t\mid h_t,x_t)\,\|\,\mathrm{sg}(p_\phi(z_t\mid h_t))\,]\big)
\end{aligned}
$$

The decoder uses symlog squared loss for vector observations, the reward head uses the
symexp two-hot loss below, and the continue predictor is logistic regression. Free bits
clip each KL below 1 nat ($\approx 1.44$ bits), disabling it once minimized so learning
focuses on prediction. The dynamics loss (stop-grad on the posterior) makes the prior
predict the representation; the representation loss (stop-grad on the prior) makes the
representation predictable, but weighted only $0.1$ so the prior does most of the moving
and the latent stays informative. Vector observations are symlog-transformed on the way
in and out. Encoder and dynamics categoricals are 1% unimix.

## Actor–critic in imagination

From replayed start states the world model and actor imagine a trajectory
$s_{1:T},a_{1:T},r_{1:T},c_{1:T}$ (imagination horizon $H=15$). The critic predicts a
*distribution* over returns; its scalar value is the mean,
$v_t=\mathbb{E}[p_\psi(\cdot\mid s_t)]$. Returns use the bootstrapped $\lambda$-return
with the continuation flag as the per-step discount, $\gamma=0.997$, $\lambda=0.95$:

$$
R^\lambda_t \doteq r_t + \gamma c_t\big((1-\lambda)v_t + \lambda R^\lambda_{t+1}\big),
\qquad R^\lambda_T \doteq v_T .
$$

**Critic.** Maximum-likelihood (symexp two-hot cross-entropy) regressing $R^\lambda_t$,
applied to imagined trajectories ($\beta_{\mathrm{val}}=1$) and replay trajectories
($\beta_{\mathrm{repval}}=0.3$). The replay value loss uses the imagination return at
the rollout start state as an on-policy bootstrap annotation, then computes
$\lambda$-returns over replay rewards. Because the target depends on the critic's own
predictions, a regularizer pulls the critic toward an EMA of its own parameters
(decay $0.98$) — a target-network effect that still lets returns use the live critic.
Output weights are initialized to zero so the agent does not hallucinate values at init.

**Actor.** Reinforce (one estimator for discrete *and* continuous actions) with a
value baseline and a fixed entropy scale $\eta=3\times10^{-4}$:

$$
\mathcal{L}(\theta)=-\textstyle\sum_t\Big[
\mathrm{sg}\!\Big(\big(R^\lambda_t-v_\psi(s_t)\big)/\max(1,S)\Big)\,
\log\pi_\theta(a_t\mid s_t)\;+\;\eta\,\mathrm{H}[\pi_\theta(a_t\mid s_t)]\Big] .
$$

Only the *range* divides the advantage (subtracting a state-independent offset leaves the
policy gradient unchanged, so the baseline handles the offset). The range is the
5th–95th percentile spread of batch returns, EMA-smoothed, and floored at $L=1$ so small
returns pass through untouched (preserving explore-when-sparse) while large returns are
compressed toward $[0,1]$:

$$
S \doteq \mathrm{EMA}\!\big(\mathrm{Per}(R^\lambda,95)-\mathrm{Per}(R^\lambda,5),\,0.99\big).
$$

## Robust prediction primitives

$$
\mathrm{symlog}(x)=\mathrm{sign}(x)\ln(|x|+1),\qquad
\mathrm{symexp}(x)=\mathrm{sign}(x)\big(\exp(|x|)-1\big),
$$

symexp is the exact inverse of symlog. Deterministic targets use
$\tfrac12(f(x)-\mathrm{symlog}(y))^2$ read out as $\hat y=\mathrm{symexp}(f)$. Stochastic
targets (reward, return) use a softmax over exponentially spaced bins
$B=\mathrm{symexp}([-20,\dots,20])$ with two-hot soft labels and cross-entropy; the
readout $\hat y=\mathrm{softmax}(f)^\top B$ can land between bins. The two-hot label of a
scalar $x$ with $k=\sum_j \delta(b_j<x)$ puts $|b_{k+1}-x|/|b_{k+1}-b_k|$ on bin $k$ and
$|b_k-x|/|b_{k+1}-b_k|$ on bin $k+1$; the loss depends only on bin *probabilities*, not
on bin values, so gradient size is decoupled from target size.

## Optimizer and architecture

Adaptive Gradient Clipping clips each tensor's gradient when it exceeds $30\%$ of the L2
norm of its weight matrix ($\epsilon=10^{-3}$), making the clip threshold scale-free.
LaProp ($\epsilon=10^{-20}$, $\beta_1=0.9$, $\beta_2=0.99$, lr $4\times10^{-5}$)
normalizes by an RMSProp denominator *then* applies momentum, permitting the tiny
epsilon. Images use stride-2 conv encoders / transposed-conv decoders (sigmoid output);
vector inputs are symlog'd then MLPs; actor and critic are 3-layer MLPs, reward and
continue 1-layer MLPs; the sequence model is a block-diagonal (8-block) GRU; RMSNorm +
SiLU throughout. The same hyperparameters hold across all benchmarks and across model
sizes from 12M to 400M parameters, with no annealing, prioritized replay, weight decay,
or dropout.

## Reference implementation sketch (PyTorch)

Core math with placeholder model APIs; this is meant to fix the indexing
and loss conventions, not to be a full production implementation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- robust scalar transform: fixed, sign-preserving, ~identity near 0 -------
def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))

# 255 raw-value bins from uniform symlog coordinates: dense near 0, huge signed reach.
SYMLIN = torch.linspace(-20.0, 20.0, 255)
BINS = symexp(SYMLIN)

def two_hot(y, bins=BINS):
    """Soft label splitting y's mass linearly across its two nearest bins."""
    bins = bins.to(device=y.device, dtype=y.dtype)
    y = torch.clamp(y, bins[0], bins[-1])
    below = (bins <= y.unsqueeze(-1)).sum(-1) - 1                  # lower bin index
    below = below.clamp(0, len(bins) - 2)                          # keep an upper neighbor
    above = below + 1
    lo, hi = bins[below], bins[above]                             # b_k, b_{k+1}
    # entry k gets |b_{k+1}-y|/|b_{k+1}-b_k|; entry k+1 gets |b_k-y|/|b_{k+1}-b_k|
    w_above = (y - lo).abs() / (hi - lo).abs()                    # weight on bin k+1
    w_below = (hi - y).abs() / (hi - lo).abs()                    # weight on bin k
    target = torch.zeros(*y.shape, len(bins), device=y.device)
    target.scatter_(-1, below.unsqueeze(-1), w_below.unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return target

def two_hot_mean(probs, bins):
    """Stable readout for bins spanning many orders of magnitude."""
    bins = bins.to(device=probs.device, dtype=probs.dtype)
    neg = bins < 0
    # Sum positive and negative contributions separately, small magnitude first.
    neg_sum = (probs[..., neg] * bins[neg]).flip(-1).sum(-1)
    pos_sum = (probs[..., ~neg] * bins[~neg]).sum(-1)
    return neg_sum + pos_sum

class TwoHotHead(nn.Module):
    """Reward predictor and critic: classifier over exp-spaced bins."""
    def __init__(self, net, bins=BINS):
        super().__init__()
        self.net = net
        self.register_buffer("bins", bins)
        nn.init.zeros_(self.net[-1].weight)                       # no value hallucination
        nn.init.zeros_(self.net[-1].bias)

    def pred(self, feat):
        return two_hot_mean(F.softmax(self.net(feat), -1), self.bins)

    def loss(self, feat, target):                                  # scale-decoupled CE
        logp = F.log_softmax(self.net(feat), -1)
        return -(two_hot(target, self.bins) * logp).sum(-1)

class SymlogMSEHead(nn.Module):                                    # decoder / deterministic
    def __init__(self, net):
        super().__init__(); self.net = net
    def pred(self, feat):  return symexp(self.net(feat))
    def loss(self, feat, target):
        return 0.5 * (self.net(feat) - symlog(target)) ** 2

def unimix(logits, alpha=0.01):                                    # 1% uniform floor
    probs = F.softmax(logits, -1)
    return (1 - alpha) * probs + alpha / probs.shape[-1]

def categorical_kl(p, q):                                          # p,q: (..., stoch, classes)
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum((-2, -1))

def straight_through_onehot(probs):
    sample = torch.distributions.OneHotCategorical(probs=probs).sample()
    return probs + (sample - probs).detach()

# --- world model loss: prediction + KL-balanced, free-bit-floored ------------
def world_model_loss(model, batch, free=1.0, beta_rep=0.1):
    post_logits, prior_logits, feat = model.observe(batch)         # RSSM forward
    post, prior = unimix(post_logits), unimix(prior_logits)
    l_pred = (model.decoder.loss(feat, batch["x"])                 # symlog MSE on symlog'd x
              + model.reward.loss(feat, batch["reward"])           # two-hot cross-entropy
              - model.cont.log_prob(feat, batch["cont"]))          # logistic regression
    kl_dyn = categorical_kl(post.detach(), prior)                  # prior chases posterior
    kl_rep = categorical_kl(post, prior.detach())                  # posterior yields, soft
    kl_dyn = torch.clamp(kl_dyn, min=free)                         # free bits: floor 1 nat
    kl_rep = torch.clamp(kl_rep, min=free)
    return (l_pred + 1.0 * kl_dyn + beta_rep * kl_rep).mean()

# --- RSSM step: deterministic GRU recurrence + categorical prior/posterior ----
class RSSMStep(nn.Module):
    def __init__(self, deter, stoch, classes, embed_dim, act_dim):
        super().__init__()
        self.stoch, self.classes = stoch, classes
        self.gru = nn.GRUCell(deter, deter)                        # block-diag GRU in practice
        self.in_proj = nn.Linear(stoch * classes + act_dim, deter) # mix latent+action -> GRU
        self.prior = nn.Linear(deter, stoch * classes)            # p(z|h)
        self.post = nn.Linear(deter + embed_dim, stoch * classes) # q(z|h,x)

    def _dist(self, logits):
        logits = logits.reshape(*logits.shape[:-1], self.stoch, self.classes)
        return unimix(logits)                                      # 1% unimix categorical

    def img_step(self, h, z, a):                                   # prior / transition
        x = torch.cat([z.flatten(-2), a], -1)
        h = self.gru(F.silu(self.in_proj(x)), h)
        prior = self._dist(self.prior(h))
        z = straight_through_onehot(prior)
        return h, z, prior

    def obs_step(self, h, z, a, embed):                            # posterior
        h, _, prior = self.img_step(h, z, a)
        post = self._dist(self.post(torch.cat([h, embed], -1)))
        z = straight_through_onehot(post)
        return h, z, post, prior

# --- lambda-return: continuation flag is the per-step discount ---------------
def lambda_return(reward, value, cont, bootstrap, lam=0.95, gamma=0.997):
    # value is aligned with reward: paper's v_t in the branch at time t.
    R = bootstrap                                                 # paper's R_T = v_T
    out = []
    for t in reversed(range(reward.shape[0])):
        R = reward[t] + gamma * cont[t] * ((1 - lam) * value[t] + lam * R)
        out.append(R)
    return torch.stack(out[::-1])

class ReturnNorm:                                                 # EMA percentile range, floor L=1
    def __init__(self, decay=0.99, limit=1.0):
        self.decay, self.limit, self.S = decay, limit, torch.tensor(0.0)
    def __call__(self, ret):
        x = ret.detach()
        self.S = self.S.to(device=x.device, dtype=x.dtype)
        lo = torch.quantile(x, 0.05); hi = torch.quantile(x, 0.95)
        self.S = (self.decay * self.S + (1 - self.decay) * (hi - lo)).detach()
        return torch.clamp(self.S, min=self.limit)                # divide only when range > 1

# --- actor + critic on an imagined rollout ----------------------------------
def actor_critic_loss(
        traj, actor, critic, slow_critic, retnorm, eta=3e-4,
        replay=None, beta_repval=0.3, slowreg=1.0):
    feat = traj["feat"]
    value = critic.pred(feat)                                     # states 0:H
    # Dreamer trajectories include the replay start state at index 0; rewards and
    # continuations at 1:H are transition outcomes for actions at 0:H-1.
    reward, cont = traj["reward"][1:], traj["cont"][1:]
    ret = lambda_return(reward, value[1:], cont, value[-1])       # returns for actions 0:H-1

    S = retnorm(ret)                                              # floored percentile range
    adv = (ret - value[:-1]) / S                                 # only divide by range

    dist = actor.dist(feat[:-1])
    logpi = dist.log_prob(traj["action"][:-1])
    # actor: Reinforce (discrete AND continuous) + fixed-scale entropy
    actor_loss = -(adv.detach() * logpi).mean() - eta * dist.entropy().mean()

    # critic: two-hot CE to lambda-return + EMA self-regularizer
    critic_loss = (critic.loss(feat[:-1], ret.detach())
                   + slowreg * critic.loss(
                       feat[:-1], slow_critic.pred(feat[:-1]).detach())).mean()
    if replay is not None:                                        # paper's replay value loss
        rfeat = replay["feat"]
        rvalue = critic.pred(rfeat)
        rret = lambda_return(
            replay["reward"][1:], rvalue[1:], replay["cont"][1:],
            replay["bootstrap"])
        critic_loss = critic_loss + beta_repval * critic.loss(
            rfeat[:-1], rret.detach()).mean()
    return actor_loss, critic_loss

# --- gradient application: scale-free clip (AGC) + LaProp --------------------
def adaptive_grad_clip(params, clip=0.3, eps=1e-3):
    for p in params:
        if p.grad is None:
            continue
        w_norm = p.detach().norm().clamp(min=eps)                # ||W||, floored
        g_norm = p.grad.norm()
        if g_norm > clip * w_norm:                               # clip per-tensor
            p.grad.mul_(clip * w_norm / (g_norm + 1e-12))
# opt = LaProp(params, lr=4e-5, betas=(0.9, 0.99), eps=1e-20)    # RMSProp-then-momentum
```
