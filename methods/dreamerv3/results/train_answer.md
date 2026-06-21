The problem is to train one reinforcement-learning agent that can be dropped into many different domains with a single fixed set of hyperparameters. The agent sees images, sensor vectors, or both, and must emit continuous or discrete actions while maximizing discounted return. What makes this hard is that the scales of rewards, returns, and observations change by orders of magnitude from one domain to another. A loss function, KL weight, or entropy coefficient that works for dense arcade scores will fail for sparse binary milestones, and a representation tuned for texture-rich 3D worlds will discard the fine pixels that matter in a static background game. Existing world-model agents such as PlaNet and earlier Dreamer generations work well after per-domain tuning, but they do not span continuous control, discrete Atari, procedural 3D worlds, and sparse open-world tasks with the same numbers.

The core insight is that almost every brittleness is a failure of scale invariance. If each objective in the agent can be rewritten so that its correct setting does not depend on the arbitrary scale or shape of rewards, returns, or observations, then one configuration can hold everywhere. The fix is not a new algorithmic skeleton but a set of robust, self-balancing primitives layered on top of the standard world-model agent: an RSSM that learns a latent dynamics model, and an actor-critic that trains entirely inside imagined rollouts produced by that model.

The method is DreamerV3. It learns a latent recurrent state-space model with a deterministic recurrent state and a stochastic categorical latent state. From replayed states, the model and actor imagine short trajectories. A critic predicts returns, and the actor is updated with a Reinforce objective. The novelty is in how predictions, dynamics regularization, return normalization, and optimization are made robust across domains.

For scalar prediction, DreamerV3 uses two complementary transforms. Deterministic targets such as vector observations are passed through symlog, a sign-preserving logarithmic transform that is approximately the identity near zero but compresses large magnitudes, and trained with squared error. The inverse symexp recovers the original scale. This prevents the squared loss from exploding on large targets without the stagnation of Huber loss or the non-stationarity of running statistics. For reward and return prediction, which can be multimodal and span huge ranges, DreamerV3 uses a symexp-twohot head. The head is a classifier over a fixed set of exponentially spaced bins covering many orders of magnitude, trained with soft two-hot cross-entropy targets. Because cross-entropy depends only on bin probabilities, the gradient size is decoupled from the target size, and the weighted-bin readout can still predict values between bins.

The world-model KL is split into two asymmetric terms. A dynamics loss, with the posterior frozen, pushes the prior to predict the representation. A representation loss, with the prior frozen, gently pushes the representation to be predictable, weighted at one tenth of the dynamics term. Both terms are floored at one nat, so once the KL is small enough the gradient is disabled and the model focuses on reconstruction instead of collapsing the latent. To prevent categorical distributions from becoming deterministic and spiking the KL, encoder, dynamics, and actor categoricals are mixed with one percent uniform mass.

For the actor, DreamerV3 uses a single Reinforce estimator for both discrete and continuous actions. Returns are normalized by the five-to-ninety-fifth percentile range of batch returns, smoothed with an exponential moving average, and the divisor is floored at one. This compresses large return magnitudes so a fixed entropy scale behaves consistently, while leaving small sparse returns untouched so exploration is preserved. The critic is trained with the same two-hot cross-entropy against lambda-returns, with an exponential moving average regularizer to stabilize bootstrapping without a frozen target network, and the output weights are initialized to zero to avoid hallucinated values at startup. The optimizer uses adaptive gradient clipping relative to each weight matrix's norm, and LaProp normalizes gradients before applying momentum.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))

SYMLIN = torch.linspace(-20.0, 20.0, 255)
BINS = symexp(SYMLIN)

def two_hot(y, bins=BINS):
    bins = bins.to(device=y.device, dtype=y.dtype)
    y = torch.clamp(y, bins[0], bins[-1])
    below = (bins <= y.unsqueeze(-1)).sum(-1) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1
    lo, hi = bins[below], bins[above]
    w_above = (y - lo).abs() / (hi - lo).abs()
    w_below = (hi - y).abs() / (hi - lo).abs()
    target = torch.zeros(*y.shape, len(bins), device=y.device)
    target.scatter_(-1, below.unsqueeze(-1), w_below.unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return target

def two_hot_mean(probs, bins=BINS):
    bins = bins.to(device=probs.device, dtype=probs.dtype)
    neg = bins < 0
    neg_sum = (probs[..., neg] * bins[neg]).flip(-1).sum(-1)
    pos_sum = (probs[..., ~neg] * bins[~neg]).sum(-1)
    return neg_sum + pos_sum

class TwoHotHead(nn.Module):
    def __init__(self, net, bins=BINS):
        super().__init__()
        self.net = net
        self.register_buffer("bins", bins)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def pred(self, feat):
        return two_hot_mean(F.softmax(self.net(feat), -1), self.bins)

    def loss(self, feat, target):
        logp = F.log_softmax(self.net(feat), -1)
        return -(two_hot(target, self.bins) * logp).sum(-1)

class SymlogMSEHead(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def pred(self, feat):
        return symexp(self.net(feat))

    def loss(self, feat, target):
        return 0.5 * (self.net(feat) - symlog(target)) ** 2

def unimix(logits, alpha=0.01):
    probs = F.softmax(logits, -1)
    return (1 - alpha) * probs + alpha / probs.shape[-1]

def categorical_kl(p, q):
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum((-2, -1))

def straight_through_onehot(probs):
    sample = torch.distributions.OneHotCategorical(probs=probs).sample()
    return probs + (sample - probs).detach()

def world_model_loss(model, batch, free=1.0, beta_rep=0.1):
    post_logits, prior_logits, feat = model.observe(batch)
    post, prior = unimix(post_logits), unimix(prior_logits)
    l_pred = (model.decoder.loss(feat, batch["x"])
              + model.reward.loss(feat, batch["reward"])
              - model.cont.log_prob(feat, batch["cont"]))
    kl_dyn = categorical_kl(post.detach(), prior)
    kl_rep = categorical_kl(post, prior.detach())
    kl_dyn = torch.clamp(kl_dyn, min=free)
    kl_rep = torch.clamp(kl_rep, min=free)
    return (l_pred + kl_dyn + beta_rep * kl_rep).mean()

class RSSMStep(nn.Module):
    def __init__(self, deter, stoch, classes, embed_dim, act_dim):
        super().__init__()
        self.stoch, self.classes = stoch, classes
        self.gru = nn.GRUCell(deter, deter)
        self.in_proj = nn.Linear(stoch * classes + act_dim, deter)
        self.prior = nn.Linear(deter, stoch * classes)
        self.post = nn.Linear(deter + embed_dim, stoch * classes)

    def _dist(self, logits):
        logits = logits.reshape(*logits.shape[:-1], self.stoch, self.classes)
        return unimix(logits)

    def img_step(self, h, z, a):
        x = torch.cat([z.flatten(-2), a], -1)
        h = self.gru(F.silu(self.in_proj(x)), h)
        prior = self._dist(self.prior(h))
        z = straight_through_onehot(prior)
        return h, z, prior

    def obs_step(self, h, z, a, embed):
        h, _, prior = self.img_step(h, z, a)
        post = self._dist(self.post(torch.cat([h, embed], -1)))
        z = straight_through_onehot(post)
        return h, z, post, prior

def lambda_return(reward, value, cont, bootstrap, lam=0.95, gamma=0.997):
    R = bootstrap
    out = []
    for t in reversed(range(reward.shape[0])):
        R = reward[t] + gamma * cont[t] * ((1 - lam) * value[t] + lam * R)
        out.append(R)
    return torch.stack(out[::-1])

class ReturnNorm:
    def __init__(self, decay=0.99, limit=1.0):
        self.decay, self.limit, self.S = decay, limit, torch.tensor(0.0)

    def __call__(self, ret):
        x = ret.detach()
        self.S = self.S.to(device=x.device, dtype=x.dtype)
        lo = torch.quantile(x, 0.05)
        hi = torch.quantile(x, 0.95)
        self.S = (self.decay * self.S + (1 - self.decay) * (hi - lo)).detach()
        return torch.clamp(self.S, min=self.limit)

def actor_critic_loss(traj, actor, critic, slow_critic, retnorm, eta=3e-4,
                      replay=None, beta_repval=0.3, slowreg=1.0):
    feat = traj["feat"]
    value = critic.pred(feat)
    reward, cont = traj["reward"][1:], traj["cont"][1:]
    ret = lambda_return(reward, value[1:], cont, value[-1])

    S = retnorm(ret)
    adv = (ret - value[:-1]) / S

    dist = actor.dist(feat[:-1])
    logpi = dist.log_prob(traj["action"][:-1])
    actor_loss = -(adv.detach() * logpi).mean() - eta * dist.entropy().mean()

    critic_loss = (critic.loss(feat[:-1], ret.detach())
                   + slowreg * critic.loss(
                       feat[:-1], slow_critic.pred(feat[:-1]).detach())).mean()
    if replay is not None:
        rfeat = replay["feat"]
        rvalue = critic.pred(rfeat)
        rret = lambda_return(
            replay["reward"][1:], rvalue[1:], replay["cont"][1:],
            replay["bootstrap"])
        critic_loss = critic_loss + beta_repval * critic.loss(
            rfeat[:-1], rret.detach()).mean()
    return actor_loss, critic_loss

def adaptive_grad_clip(params, clip=0.3, eps=1e-3):
    for p in params:
        if p.grad is None:
            continue
        w_norm = p.detach().norm().clamp(min=eps)
        g_norm = p.grad.norm()
        if g_norm > clip * w_norm:
            p.grad.mul_(clip * w_norm / (g_norm + 1e-12))
# opt = LaProp(params, lr=4e-5, betas=(0.9, 0.99), eps=1e-20)
```
