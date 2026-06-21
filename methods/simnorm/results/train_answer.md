I am training a model-based agent that learns an *implicit* world model entirely in latent space and plans there. An encoder $h$ maps an observation $s$ to a latent $z$, a dynamics network predicts the next latent $z' = d(z,a)$, and reward and value heads read off $z$ — but there is no decoder, nothing that reconstructs the observation to ground $z$. The model is trained by a self-predictive consistency loss that pulls the predicted next latent toward the encoder's own stop-gradient embedding of the next observation, $\|d(z_t,a_t) - \operatorname{sg}(h(s_{t+1}))\|^2$, plus reward regression and a bootstrapped value loss whose target $q = r + \gamma \bar Q(z', p(z'))$ is itself a function of latents the same network produces. So $z$ is defined purely by these self-referential objectives, and that is exactly where it hurts: on harder continuous-control tasks the gradient norm climbs over training and learning diverges. Before reaching for a fix I want to know *why*, because clipping gradients would only treat a symptom.

The diagnosis is a missing degree of freedom. Read the consistency loss as an optimizer would: it wants $d(z,a)$ to match $\operatorname{sg}(h(s'))$, and a cheap way to shrink a squared error in feature space — with nothing to do with learning real dynamics — is to shrink the whole representation. The bootstrapped value loop pushes the other way: the target $\bar Q(z',\cdot)$ reads the very $z'$ the dynamics emits, so latents feed value targets, value gradients feed back into $h$ and $d$, which reshape the latents. That is a closed loop, and a loop with an unconstrained magnitude variable has a runaway direction. The overall *scale* of $z$ is a degree of freedom the losses never pin down, and the feedback can drive it — the exploding gradients are that magnitude becoming uncontrolled. The fix must remove the degree of freedom by bounding the representation. The existing options each fall short of doing so cleanly. My layers already use LayerNorm, but it normalizes the *intermediate* pre-activations flowing through each layer; the final $z$ is produced by a last linear map after the last LayerNorm, with arbitrarily large weights — LayerNorm controls the river, not the mouth, so the thing I actually consume stays unbounded. Plain L2-normalization, $z \leftarrow z/\|z\|_2$, kills the magnitude degree of freedom dead, but the sphere bounds magnitude and *nothing else*: every point is equally available, one global division couples all coordinates through a single scalar, and the code stays dense and shapeless. A squashing tanh bounds each coordinate but creates no competition between coordinates, leaves the code dense, and — worse for me — saturated units pass almost no gradient, trading exploding gradients for dead ones. A hard VQ-style code has the geometry I want but is non-differentiable.

I propose **SimNorm** — simplicial normalization — a final activation that defines the geometry of $z$ itself by treating the latent as a collection of independent simplices. Let me build it from the geometry I actually want rather than from a list of normalizers. There is an old intuition I trust: sparse, overcomplete codes — more components than the input dimension, most inactive — are stable under noise and are the representations downstream linear readouts like, and my reward and value heads *are* linear readouts on $z$. The VQ-VAE picture of a *vector of categoricals* — split the latent into $L$ groups and replace each by a one-hot code — has exactly the right shape: it is bounded (one-hots have unit norm), maximally sparse within each group, and overcomplete, since $L$ groups with $V$ choices each represent up to $V^L$ states, on the order of $L\log_2 V$ bits, giving expressivity and boundedness at once. The trouble is only that the one-hot is an $\operatorname{argmax}$, non-differentiable, dragging in a straight-through estimator, a codebook, a commitment loss, and codebook-collapse pathologies — fragile machinery for a recurrent model where gradients must flow cleanly through $d$ step after step. So I keep the group geometry but make it *soft*. The smooth relaxation of $\operatorname{argmax}$ is softmax. Reshape $z$ from $(\dots,d)$ to $(\dots,L,V)$, apply softmax within each group of $V$ entries, and reshape back:

$$z^\circ = [\,g_1, \dots, g_L\,], \qquad g_i = \operatorname{softmax}(z_{(i)}/\tau), \qquad g_{ij} = \frac{e^{z_{ij}/\tau}}{\sum_k e^{z_{ik}/\tau}}.$$

Each group $g_i$ is nonnegative and sums to $1$ — a point on the $(V-1)$-simplex — and concatenating the $L$ of them gives $z^\circ$.

This bounds the representation structurally, which was the whole point. Within a group, $\|g_i\|_1 = 1$ exactly, and since each entry lies in $[0,1]$, $\|g_i\|_2^2 = \sum_j g_{ij}^2 \le \sum_j g_{ij} = 1$, so $\|g_i\|_2 \le 1$. Across all $L$ groups, $\|z^\circ\|_1 = \sum_i \|g_i\|_1 = L$ exactly, and $\|z^\circ\|_2^2 = \sum_i \|g_i\|_2^2 \le L$, so $\|z^\circ\|_2 \le \sqrt{L}$, with the matching lower bound $\sqrt{L/V} \le \|z^\circ\|_2$ since each group's mass is at least $1/V$ in $\ell_2$. The upper bound depends on *nothing about the input or task*, only on $L$ — exactly the degree of freedom I diagnosed as the source of the blowup, removed with no scalar constant to tune per task.

The group structure is not optional, and it is worth seeing why a single global softmax over all $d$ entries is the wrong choice. One softmax over $d$ forces $\sum z^\circ = 1$ across the whole vector, so on average each coordinate is about $1/d$ — vanishing for $d$ in the hundreds — only one coordinate can be appreciably active at once, and the code carries on the order of $\log_2 d$ bits: a brutal bottleneck for a latent that must encode the full state for dynamics, reward, and value. Splitting into $L$ independent simplices fixes precisely this. Each group independently chooses which of its $V$ entries to activate, giving $L$ near-independent decisions, an overcomplete code with up to $V^L$ configurations and roughly $L\log_2 V$ bits, while every group stays individually bounded. Boundedness and expressivity stop fighting.

The sparsity I wanted arrives as a *bias*, not an imposed constraint, through the mechanism of the softmax itself. Within a group the outputs sum to $1$ and are all nonnegative, so it is a zero-sum competition: to raise one component by $\alpha$ the rest must collectively give up $\alpha$. To drive the consistency and value losses down the network cannot make everything large; it must prioritize, pushing mass onto a few entries per group at the expense of the rest, so the learned $g_i$ drift toward approximately one-hot — sparse but soft — with no L1 term and no hard $\operatorname{argmax}$. Per-group entropy $H(g_i) = -\sum_j g_{ij}\log g_{ij} \in [0, \ln V]$ quantifies it: $0$ is one-hot (maximally sparse), $\ln V$ is uniform (maximally dense), and the representation settles in between, biased toward the sparse end.

Two knobs need to be set on purpose. The temperature is the dial: as $\tau \to 0^+$ the softmax sharpens to a hard $\operatorname{argmax}$ and each group becomes one-hot — the exact discrete vector-of-categoricals, now reached continuously and differentiably; as $\tau \to \infty$ every logit washes out, each group goes uniform at $1/V$, carrying no information and blocking gradient flow. Very small $\tau$ is dangerous here for the same reason vanishing gradients are: near one-hot the softmax Jacobian collapses and almost no gradient reaches the encoder and dynamics, and clean gradient flow through $d$ over a multi-step rollout is the whole game. Since the point of this exercise was stability *without* per-task tuning, I take the neutral default $\tau = 1$, a plain softmax, which keeps a healthy sparsity bias while leaving gradients flowing. The group size $V$ trades stronger sparsity pressure and more bits per group (larger $V$) against cheaper, sharper-gradient simplices (smaller $V$); with a latent dimension in the hundreds I want enough groups $L = d/V$ to keep the code rich and a small $V$ to keep each simplex cheap and gradient-friendly, so I take $V = 8$ — eight-way competition per group, $L = d/8$ groups (64 simplices at $d = 512$) — which also keeps the reshape clean, needing only $V \mid d$.

Finally, *where* it goes. The consistency loss compares $d(z,a)$ against $\operatorname{sg}(h(s'))$, so for that comparison to be meaningful the prediction and the target must live in the same space — same geometry, same bound. SimNorm therefore goes on the final activation of *both* the encoder $h$ and the dynamics $d$ (every encoder path, state and RGB alike). If I normalized only the encoder, the dynamics could emit an unbounded $z'$ and I would reintroduce the very magnitude drift I removed, plus a geometry mismatch between predictor and target. The reward, policy, and Q heads then always consume a bounded, sparse, simplex-valued $z$, which keeps their inputs well-scaled and their Jacobians tame. The layer itself is just a reshape, a softmax over the last axis, and a reshape back — shape-preserving, so it drops straight into the `mlp(..., act=...)` slot.

```python
import torch.nn as nn
import torch.nn.functional as F

class SimNorm(nn.Module):
    """Simplicial Normalization. Partition the input into groups of size `dim` (V)
    and softmax each group, projecting it onto a simplex. Shape-preserving."""
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim          # V; default 8

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim) # (..., d) -> (..., L, V)
        x = F.softmax(x, dim=-1)            # softmax within each group -> point on a simplex
        return x.view(*shp)                 # back to (..., d)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"
```

It is the final activation of every encoder path and the latent dynamics, not the reward, policy, or Q heads:

```python
# state encoder; task_dim is 0 in single-task runs
out["state"] = mlp(cfg.obs_shape["state"][0] + cfg.task_dim,
                   max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                   cfg.latent_dim, act=SimNorm(cfg))
# RGB encoder path also ends with SimNorm
out["rgb"] = conv(cfg.obs_shape["rgb"], cfg.num_channels, act=SimNorm(cfg))
# dynamics prediction lives in the same bounded space as sg(h(s'))
self._dynamics = mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                     2 * [cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg))
```
