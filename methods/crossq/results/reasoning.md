Let me start from a thing everyone in off-policy continuous control treats as furniture: the target network. SAC, TD3, DDPG all do the same move when they form the Bellman target. To regress the live critic `Q_θ(s,a)` toward `r + γ V(s')`, the value of the next state is read off a *slowly tracked copy* of the critic, `Q_{θ̄}`, with `θ̄ ← (1−τ)θ̄ + τθ` and τ tiny. The justification is the DQN one: a deep critic needs many gradient steps to fit a target, and if the target is computed from the live network then every step both chases the target and moves it, leaving residual error the next step inherits and compounds. Freeze the target, fit a stationary objective, no self-chasing. It works, and it is everywhere. But look at what it costs me. The target copy is *deliberately stale* — by construction it lags the live critic — so the value signal driving learning is always behind, and τ is one more knob to balance: too large and the target jitters, too small and learning crawls. The high-UTD methods that get genuinely sample-efficient, REDQ and DroQ, do not remove this; they pour gradient steps and ensembles on top of it. So the sample-efficiency frontier is being bought with computation, and the staleness is still in there. The question I want to actually ask is the one nobody asks because the answer seems obvious: is the target network *necessary*, or is it patching a problem whose real cause is somewhere else?

Let me re-derive *why* bootstrapping diverges without a target, carefully, because the folklore answer ("the target moves") is not specific enough to attack. The critic learns by minimizing `(Q_θ(s,a) − [r + γ Q_θ(s',a')])²` where `a' ∼ π(·|s')`. If I drop the target copy and use the live `Q_θ` at both `(s,a)` and `(s',a')`, the gradient through the second term pushes `Q_θ(s',a')` *down* as well (the optimizer will happily shrink the target to shrink the loss), which is a degenerate signal, and the two evaluations interact. Standard practice already half-fixes the degeneracy by stop-gradient on the target: detach `Q_θ(s',a')` so only the prediction at `(s,a)` gets a gradient. So suppose I do that — detach the next-state value, no gradient through it — and still use the *live* network to compute it. Now there is no degenerate "shrink the target" path. What's left that diverges? Two things. One, the target value still moves every step because θ moves, so the regression objective is non-stationary — the DQN story. Two, and this is the one I think is actually doing the damage, the distribution the critic is evaluated on at `(s,a)` and at `(s',a')` are *not the same distribution*, and a deep net's outputs on out-of-distribution inputs are unconstrained. The actions `a` at the current state come from the replay buffer — a mixture of stale policies. The actions `a'` at the next state come from the *current* policy `π`. Those are different action distributions over different states, and the critic is being asked to be simultaneously accurate on both and to relate them through the Bellman equation. If the critic's behavior on the `(s',a')` cloud drifts away from its behavior on the `(s,a)` cloud — which nothing prevents, because they are different regions of input space — the bootstrap relates two numbers that live on different scales, and the recursion amplifies the mismatch. The target network masks this second problem by freezing one side, but it does not address it; it just slows everything down enough that the mismatch can't run away.

That reframing is the lever. If the real disease is a *distribution mismatch between the current and next state-action inputs to the same critic*, then the cure is not "freeze one side" — it's "make the two sides share a normalized distribution so the critic sees them as one population." And there is a standard tool whose entire job is to normalize a layer's inputs to a controlled distribution: Batch Normalization. So the move I want to try is: put BatchNorm inside the critic, and normalize away the very mismatch that the target network was compensating for. If that works, the target network becomes redundant and I can delete it.

But BatchNorm has a bad reputation in RL, and I should not wave that away — I should reproduce the failure so I know what I'm fixing. Drop a vanilla BatchNorm into the critic the obvious way: forward the current batch `(s,a)` through `Q_θ` to get the prediction, separately forward the next batch `(s',a')` through `Q_θ` (under no-grad) to get the bootstrap value, form the target, regress. Now think about what BatchNorm computes in each of those two forward passes. In the first pass it normalizes by the mean/variance of the `(s,a)` batch. In the second pass it normalizes by the mean/variance of the `(s',a')` batch. Those are *different* normalization statistics, because — exactly as I argued — the two batches have different statistics (different states, and current-policy actions vs. replay actions). So the critic is literally a *different function* in the two passes: the same weights but different normalization, hence different effective layer transforms. The Bellman equation `Q(s,a) = r + γ Q(s',a')` now relates the outputs of two non-identical functions, which is incoherent — I've made the mismatch worse, not better, and the running statistics, updated separately by each pass, lurch around. That is precisely why naive BatchNorm destabilizes value learning: TD relates predictions from two forward passes that BatchNorm has normalized under two different moments. Good — the failure has a specific, fixable cause.

The fix writes itself once the cause is named. I must not let BatchNorm see the two batches as two different populations. So *concatenate* them: stack `(s,a)` on top of `(s',a')` into one batch of size `2N`, do a *single* forward pass through the BatchNorm critic, then split the output back into the current-state predictions and the next-state values. Now BatchNorm computes its normalization moments from the *union* of both sub-batches — one shared distribution — so every input, current or next, is normalized identically, and the critic is one consistent function across both. The Bellman target and the prediction now live on the same normalized scale by construction. This is the whole idea, and it is almost free: one `torch.cat`, one forward pass (which is also cheaper than two separate passes), one `torch.split`. The next-state half of the output is detached before forming the target — it's a bootstrap value, no gradient flows into it — but it shares the same forward pass and therefore the same normalization as the current-state half. That shared normalization is what the target network was clumsily approximating, and now I get it exactly. So: BatchNorm in the critic, current and next forwarded jointly, and — the payoff — *no target network at all*. The bootstrap value comes from the live critic, normalized jointly with the prediction.

I want to make sure I believe the "no target network" claim and am not just hoping. The reason a live-critic bootstrap diverged was (a) non-stationarity and (b) the cross-batch distribution mismatch. The joint-forward BatchNorm directly kills (b): both sides share normalization, so the critic can't drift to different scales on the two clouds. And it substantially tames (a): BatchNorm fixes the *scale and center* of every layer's activations to a controlled distribution every step, so even though θ moves, the normalized representation the later layers see is far more stationary than raw activations would be — the network's internal targets don't wander the way they do without normalization. So the two roles the target network was playing — stationarity and cross-batch consistency — are both taken over by the joint-batch normalization, and the target network has nothing left to do. Delete it.

Now the reputation problem comes back in a subtler form, and it's the reason plain BatchNorm still isn't quite enough. BatchNorm normalizes by *batch* statistics during training but by *running* statistics at inference, and that train/inference gap is fine in supervised learning where batches are large and i.i.d. In RL the data is non-stationary — the policy is changing, so the replay distribution drifts, the running statistics are always chasing a moving target, and the batch statistics on a given minibatch can swing. When the running stats and the current batch stats disagree a lot, the network at "inference-style" usage behaves differently from training, and the value estimates wobble. Batch Renormalization (Ioffe 2017) is exactly the patch for this: it keeps normalizing by batch statistics but adds an affine correction `(r, d)` that ties the batch normalization back toward the running statistics, with `r` and `d` *clipped* to bounded ranges `[1/r_max, r_max]` and `[−d_max, d_max]` and treated as constants (stop-gradient). Early in training, `r_max=1, d_max=0` makes it identical to plain BatchNorm; the clip ranges are then widened over a warmup so the correction can engage. The effect is that training-time and running-time normalization are kept consistent, which is the robustness I need under non-stationary RL data. So the critic's normalization layer is BatchRenorm, not BatchNorm — same joint-forward usage, but with the clipped correction that survives drifting batch statistics. The running statistics update with a small momentum (0.01 on the new batch), and the correction terms are detached so they shape the normalization without being differentiated through.

That's the core. Everything else I keep from SAC unchanged, deliberately, because the contribution is supposed to be a few lines on top of SAC and I don't want to confound it. The actor is the same stochastic tanh-Gaussian, sampled by reparameterization, with the change-of-variables log-prob correction. The objective is the same maximum-entropy one, with automatic temperature tuning: keep a learned `log α`, descend `−log α · (log π(a|s) + H_target)` with `H_target = −dim(A)`, so α self-adjusts to hold the policy's entropy near target. I keep *twin* critics and take the *min* of the two next-state Q-values before subtracting the entropy term — the clipped-double-Q guard against overestimation is orthogonal to the normalization change and still wanted, since the actor still ascends the critic and still chases whatever the critic overrates. So the next-state value is `min(Q1(s',a'), Q2(s',a')) − α log π(a'|s')`, both critics forwarded jointly with their current-state inputs, the min detached, the TD target `r + γ(1−d)·that`.

A few consequences of removing the target network are worth stating because they change the hyperparameters. First, there is no τ anymore — the soft-update coefficient is gone, one fewer knob. Second, because the critic now learns from a *live, un-stale* bootstrap and the normalization keeps it stable, the critic learns faster, and I can update the actor aggressively: UTD stays at 1 (one critic update per environment step, no ensemble, no extra gradient steps), and the actor is updated frequently — the paper updates the actor every few critic steps (a small policy delay), not heavily delayed. Third, normalization makes the critic robust enough that I can *widen* it: the paper uses much wider critic hidden layers (2048 units) than SAC's 256, which the normalization makes trainable and which buys accuracy. The actor stays narrow. So the recipe is: wide BatchRenorm twin critics, no targets, joint current+next forward pass, SAC actor and entropy tuning, UTD = 1.

Let me assemble the loop. Per environment step: act with the stochastic actor, store the transition. Then one update: sample a minibatch `(s,a,r,s',d)`; sample `a' ∼ π(s')` and its log-prob with no grad; put both critics in train mode and forward `cat[(s,a),(s',a')]` through each, split into current and next halves; detach the next halves, take their elementwise min, subtract `α·log π(a'|s')`, form `target = r + γ(1−d)·that`; the critic loss is `MSE(Q1_cur, target) + MSE(Q2_cur, target)`, step it. Then (every policy-delay steps) put the critics in eval mode, sample `ã ∼ π(s)`, compute `min(Q1(s,ã), Q2(s,ã))`, and step the actor on `(α·log π(ã|s) − minQ).mean()`; update α on `−log α·(log π + H_target)`. No target sync anywhere. The critic-train/critic-eval toggle around the actor update matters: the actor reads the critic at the *current* batch only, so it should use running statistics (eval mode), not recompute batch stats on a single-distribution batch.

Here is the core, with the BatchRenorm critic, the joint forward pass, and the target-network-free update.

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class BatchRenorm1d(nn.Module):
    # Batch Renormalization (Ioffe 2017): batch-norm with clipped corrections
    # (r, d) that tie batch stats to running stats; robust to non-stationary RL data.
    def __init__(self, num_features, momentum=0.01, eps=1e-3, rmax=3.0, dmax=5.0):
        super().__init__()
        self.momentum, self.eps, self.rmax, self.dmax = momentum, eps, rmax, dmax
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            bmean = x.mean(0)
            bvar = x.var(0, unbiased=False)
            bstd = (bvar + self.eps).sqrt()
            rstd = (self.running_var + self.eps).sqrt()
            r = (bstd / rstd).detach().clamp(1.0 / self.rmax, self.rmax)
            d = ((bmean - self.running_mean) / rstd).detach().clamp(-self.dmax, self.dmax)
            xhat = (x - bmean) / bstd * r + d
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * bmean.detach())
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * bvar.detach())
        else:
            xhat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        return self.weight * xhat + self.bias


class Critic(nn.Module):
    # wide BatchRenorm Q-network; no target copy exists for this net
    def __init__(self, obs_dim, act_dim, hidden=2048):
        super().__init__()
        self.l1 = nn.Linear(obs_dim + act_dim, hidden); self.bn1 = BatchRenorm1d(hidden)
        self.l2 = nn.Linear(hidden, hidden); self.bn2 = BatchRenorm1d(hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        return self.l3(x).view(-1)


class Actor(nn.Module):
    # SAC stochastic tanh-Gaussian actor, unchanged
    def __init__(self, obs_dim, act_dim, max_action, hidden=256):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.max_action = max_action

    def forward(self, s):
        x = F.relu(self.l1(s)); x = F.relu(self.l2(x))
        log_std = torch.tanh(self.log_std(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return self.mu(x), log_std

    def sample(self, s):
        mu, log_std = self(s)
        normal = torch.distributions.Normal(mu, log_std.exp())
        u = normal.rsample()
        y = torch.tanh(u)
        a = y * self.max_action
        logp = normal.log_prob(u) - torch.log(self.max_action * (1 - y.pow(2)) + 1e-6)
        return a, logp.sum(1, keepdim=True)


class CrossQ:
    def __init__(self, obs_dim, act_dim, max_action, device,
                 gamma=0.99, lr=1e-3, policy_delay=3):
        self.device, self.gamma, self.policy_delay = device, gamma, policy_delay
        self.max_action, self.total_it = max_action, 0
        self.actor = Actor(obs_dim, act_dim, max_action).to(device)
        self.qf1 = Critic(obs_dim, act_dim).to(device)
        self.qf2 = Critic(obs_dim, act_dim).to(device)  # twin critics, NO targets
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr)
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.al_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    def _joint(self, qf, s, a, s2, a2):
        # current and next forwarded TOGETHER so BatchRenorm shares one distribution
        q = qf(torch.cat([s, s2], 0), torch.cat([a, a2], 0))
        return q[: s.shape[0]], q[s.shape[0]:]

    def update(self, batch):
        self.total_it += 1
        s, s2, a, r, d = batch
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
        self.qf1.train(); self.qf2.train()
        q1_cur, q1_nxt = self._joint(self.qf1, s, a, s2, a2)
        q2_cur, q2_nxt = self._joint(self.qf2, s, a, s2, a2)
        with torch.no_grad():
            min_nxt = torch.min(q1_nxt.detach(), q2_nxt.detach()) - self.alpha * logp2.view(-1)
            target = r + (1 - d) * self.gamma * min_nxt
        q_loss = F.mse_loss(q1_cur, target) + F.mse_loss(q2_cur, target)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        if self.total_it % self.policy_delay == 0:
            self.qf1.eval(); self.qf2.eval()
            pi, logp = self.actor.sample(s)
            min_pi = torch.min(self.qf1(s, pi), self.qf2(s, pi))
            a_loss = (self.alpha * logp.view(-1) - min_pi).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            alpha_loss = (-self.log_alpha.exp() * (logp.detach().view(-1) + self.target_entropy)).mean()
            self.al_opt.zero_grad(); alpha_loss.backward(); self.al_opt.step()
            self.alpha = self.log_alpha.exp().item()
        return {"q_loss": q_loss.item()}
```

The causal chain, start to finish: the target network is universally assumed necessary for off-policy value bootstrapping, but its real job turns out to be compensating for a *distribution mismatch* between the current state-action batch (replay actions) and the next state-action batch (current-policy actions), which a single shared critic evaluates on two different input clouds; Batch Normalization is the natural tool to remove that mismatch, but naive insertion makes it worse because it normalizes the two batches under *different* moments — so the Bellman equation relates two non-identical functions; concatenating the current and next batches into one joint forward pass forces both to be normalized under one shared distribution, which is the consistency the target network was approximating and now I get exactly; Batch Renormalization, not plain BatchNorm, supplies the train/inference consistency needed under non-stationary RL data; together they let me delete the target network outright, keep UTD at 1, widen the critic, and otherwise leave SAC — stochastic actor, twin-min critics, entropy auto-tuning — untouched.
