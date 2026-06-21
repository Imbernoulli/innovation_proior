# Value Iteration Networks (VIN)

## Problem

Reactive deep policies (a perception CNN mapping an observation straight to an action distribution) fail to generalize across instances of a task that requires planning — e.g. grid-world navigation with new obstacle/goal layouts. The tell-tale diagnostic: on held-out maps the per-step action-prediction loss stays comparable to a planner's, but trajectory success rate is far lower and the gap grows with map size. Reactive nets learn the locally easy parts of trajectories and miss the globally goal-coupled parts, because nothing in their forward pass performs lookahead; generalizing would require memorizing astronomically many layouts.

## Key idea

Embed a **differentiable planner** inside the policy. Posit a latent planning MDP $\bar M$ whose reward $\bar R=f_R(\phi(s))$ and transitions $\bar P=f_P(\phi(s))$ are *learned from the observation* (a useful caricature, not the true model). Plan in $\bar M$ by value iteration to get the value field $\bar V^*$, then react to the local slice of it at the current state. The enabling observation: **one value-iteration sweep on a grid is a convolution followed by a channel-wise max**, so $K$ sweeps are a recurrent convolutional module that is fully differentiable and trains end-to-end by backpropagation with any standard RL/IL algorithm.

## The VI module (value iteration as a CNN)

Value iteration is
$$Q_n(\bar s,\bar a)=\bar R(\bar s,\bar a)+\gamma\sum_{\bar s'}\bar P(\bar s'\mid\bar s,\bar a)V_n(\bar s'),\qquad V_{n+1}(\bar s)=\max_{\bar a}Q_n(\bar s,\bar a).$$
On a 2-D grid with local transitions, $\sum_{\bar s'}\bar P(\bar s'\mid\bar s,\bar a)V_n(\bar s')$ is a **convolution** of the value image $V_n$ with a kernel equal to the (discounted) transition probabilities of action $\bar a$; adding the reward gives the $Q$ map for $\bar a$. Stacking one such convolution channel per action makes the whole $Q$ tensor one convolution layer (each channel $=Q(\cdot,\bar a)$, kernel weights $=$ discounted transitions). The Bellman $\max_a$ is **max-pooling over the action (channel) axis**, collapsing the $Q$ channels to the next value image $V_{n+1}$. Feeding $V_{n+1}$ (stacked with the reward) back through the same conv-and-channel-max, $K$ times, performs $K$ VI sweeps.

Design choices and why:
- **Tied weights across the $K$ sweeps** — value iteration applies the *same* Bellman operator every sweep; tying gives effective depth $K$ with one sweep's parameters and is the inductive bias that makes "learn to plan" data-efficient. Untying the weights degrades performance, most at small data sizes.
- **$3\times3$ transition kernels** — grid transitions are local; small kernels capture them with minimal parameters.
- **$K\propto$ map diameter** — each sweep propagates value one cell; need $K$ large enough that goal information reaches every state (e.g. $K=10,20,36$ for $8\times8,16\times16,28\times28$). Too small ⇒ distant states are goal-blind; too large ⇒ costly and harder to train.
- **A few extra $Q$ channels** (e.g. $l_q=10$ for 8 actions) — the learned planner's latent action set need not match the true one; extra channels give it more freedom and tend to help.
- **Attention = index the local $Q$** — by Bellman, the optimal action needs only the value of the current state's reachable neighbors; reading off the $Q$-vector at the current cell (soft-attention, Xu et al. 2015) cuts effective parameters and speeds learning.
- **$f_R$ is a small conv net** — it learns a reward-shaping (negative at obstacles, positive at goal, small negative elsewhere); transitions are learned $3\times3$ kernels independent of the observation. The network is never told the true model yet plans successfully.

## Algorithm

1. $\bar R = f_R(\phi(s))$: conv($3\times3$, $l_h$ channels) → conv(→ 1 channel) on the observation image.
2. $\bar Q = \mathrm{conv}_{3\times3}(\bar R)$ with $l_q$ action channels; $\bar V = \max_{\text{channel}}\bar Q$.
3. Repeat $K-1$ times: $\bar Q = \mathrm{conv}_{3\times3}([\bar R,\bar V];\,[W_R,W_{\bar P}])$; $\bar V=\max_{\text{channel}}\bar Q$. Then one final backup. ($W_R,W_{\bar P}$ tied across sweeps.)
4. Attention: $\psi(s)=\bar Q[:,:,i_s,j_s]$, the $Q$-vector at the current cell.
5. Reactive readout: logits $=W_o\,\psi(s)$; $\pi=\mathrm{softmax}$.

Train by cross-entropy against expert actions (imitation) or any RL loss (e.g. TRPO), with RMSProp. The whole policy is $\pi_\theta(a\mid\phi(s))$ and trains end-to-end.

## Code (faithful to the canonical implementation)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class VIN(nn.Module):
    """Value Iteration Network: a policy with an embedded differentiable planner.
    Learns a latent reward map, runs K value-iteration sweeps as a recurrent conv
    module, attends to the local Q at the current state, and reacts."""

    def __init__(self, in_channels=2, l_h=150, l_q=10, num_actions=8):
        super().__init__()
        # f_R : observation image (obstacle map, goal map) -> reward map.
        self.h = nn.Conv2d(in_channels, l_h, kernel_size=3, stride=1, padding=1, bias=True)
        self.r = nn.Conv2d(l_h, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # Reward -> Q weights (one 3x3 kernel per latent action): the R(.,a) term.
        self.q = nn.Conv2d(1, l_q, kernel_size=3, stride=1, padding=1, bias=False)
        # Value-feedback -> Q weights (gamma * P(.|.,a)); tied across all K sweeps.
        self.w = Parameter(torch.zeros(l_q, 1, 3, 3))
        # Reactive readout: local Q-vector -> action logits.
        self.fc = nn.Linear(l_q, num_actions, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, obs_image, state_x, state_y, k):
        h = self.h(obs_image)
        r = self.r(h)                                   # latent reward map (1 channel)

        q = self.q(r)                                   # initial Q from reward alone
        v, _ = torch.max(q, dim=1, keepdim=True)        # V = max_a Q (max over action channels)

        def bellman(r, v):
            # one VI sweep as a conv over the [reward, value] stack:
            # reward-side kernels (R) concat value-feedback kernels (gamma*P).
            return F.conv2d(torch.cat([r, v], dim=1),
                            torch.cat([self.q.weight, self.w], dim=1),
                            stride=1, padding=1)

        for _ in range(k - 1):                          # K-1 more tied Bellman backups
            q = bellman(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)
        q = bellman(r, v)                               # final sweep's Q

        # attention: index Q at the current cell (locality of the Bellman backup).
        batch_sz, l_q = q.size(0), q.size(1)
        q_local = q[torch.arange(batch_sz), :, state_x.long(), state_y.long()].view(batch_sz, l_q)

        logits = self.fc(q_local)                       # reactive policy on attended Q
        return logits, self.sm(logits)


def train(model, loader, epochs, k, lr=0.01):
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    for _ in range(epochs):
        for obs_image, state_x, state_y, action in loader:
            logits, _ = model(obs_image, state_x, state_y, k)
            loss = F.cross_entropy(logits, action)      # imitation; swap for an RL loss if desired
            opt.zero_grad(); loss.backward(); opt.step()
```

Notes faithful to the canonical version: input is a 2-channel image (obstacle map, goal map) plus the agent's $(i,j)$ cell; $l_h=150$, $l_q=10$, 8 action outputs; the convolution is padded to keep the value image at full resolution; $K$ is set by map size ($10/20/36$ for $8/16/28$). The whole network is tiny (on the order of a couple thousand parameters). For a graph domain (e.g. web navigation), the same construction uses graph-structured transitions, a learned per-node reward from the query, and an attention that reads values over relevant nodes; for continuous control, attention selects a small patch of the value map around the discretized position and an FC head outputs continuous controls.
