We want a neural-network policy for sequential decision-making — navigation, control, trajectory prediction — that generalizes to task instances it was never trained on: train on many random grid-world maps with scattered obstacles and a random goal, then solve a *new* layout drawn from the same distribution. The standard recipe is a reactive deep policy: a perception CNN extracts features $\phi(s)$ from the observation, fully connected layers map them to a softmax over actions, and the whole thing $\pi_\theta(a\mid\phi(s))$ is trained by imitation (cross-entropy against expert actions) or by model-free RL. On the maps it trained on it is fine; on a held-out map it falls apart. And the *way* it falls apart is the clue to what to build. The per-step action-prediction loss on test maps stays comparable to a planner's — yet the trajectory success rate is far lower, and the gap widens sharply as the map grows, from passable at $8\times8$ to hopeless at $28\times28$, while a real planner barely notices the size change. This is not ordinary overfitting; if it were, the per-step loss would show it. The reactive net is nailing the locally easy steps (when there is a clear shot at the goal, head that way) and blowing the decisive steps that require reasoning about the whole layout (when you must go *away* from the goal to round a wall). On a big map one such miss kills the trajectory, and there are many of them. The deeper reason is that a reactive net is a fixed feedforward function with no point in its computation where it looks ahead, so it can only succeed by memorizing, across training maps, a lookup of "local obstacle pattern and goal-relative position $\Rightarrow$ action." With $K$ obstacles on an $n\times n$ grid there are on the order of $\binom{n^2}{K}$ layouts, each demanding a different optimal action field — memorization is hopeless. The one thing that *would* transfer across all of them is the planning computation itself, which is the same algorithm on every map.

So I do not want a better-tuned reactive net; I want the policy to *contain a planner*, so that solving a new map is just running the same computation on new inputs. The obvious route, model-based RL — identify the dynamics, then solve the MDP — generalizes in principle but stakes everything on system identification, and the map from model to optimal plan is so sensitive that a small modelling error can flip the plan entirely, which is exactly why model-free methods are trusted in practice. Inverse RL and max-margin planning are no better for my purpose: they learn a cost but assume the transition model is *known* and call a separate black-box planner outside the network. I want everything to be one differentiable net — learning its own reward *and* transitions — trained by the same RL/IL machinery as a reactive policy.

I propose the **Value Iteration Network (VIN)**: a policy with a differentiable planner embedded directly in its forward pass. The construction is to posit a latent planning MDP $\bar M$ that I get to invent — I do not claim it is the *true* MDP of the world, only that some useful caricature exists — whose reward and transitions are learned from the observation, $\bar R = f_R(\phi(s))$ and $\bar P = f_P(\phi(s))$. The policy plans in $\bar M$ by value iteration to obtain the optimal value field $\bar V^*$, then reacts to a local slice of that field at the current state. Why hand $\bar V^*$ to the reactive readout? Because the value function is a sufficient statistic of the plan: once $\bar V^*$ is known over all states, the optimal action anywhere is $\arg\max_{\bar a}\,\bar R(\bar s,\bar a)+\gamma\sum_{\bar s'}\bar P(\bar s'\mid\bar s,\bar a)\bar V^*(\bar s')$, so $\bar V^*$ encodes the entire optimal behavior. Staring at that argmax reveals more: the optimal action at $\bar s$ depends only on $\bar V^*$ at the states reachable from $\bar s$, i.e. with $\bar P(\bar s'\mid\bar s,\bar a)>0$. On a grid, transitions are local — from a cell you can only step to its few neighbors — so the action at a cell depends only on the value of a tiny neighborhood. That is exactly the soft-attention idea (Xu et al. 2015): for any one prediction, only a subset of inputs is relevant, so route only that subset to the predictor, cutting the effective parameter count and speeding learning. I therefore insert an attention module that selects the local value slice $\psi(s)$ at the current state and feeds only it to a small reactive readout $\pi_{\text{re}}(a\mid\phi(s),\psi(s))$. Since $\psi(s)$ is ultimately a function of $\phi(s)$, the whole thing is still an ordinary parametrized policy $\pi_\theta(a\mid\phi(s))$, trainable by any algorithm that needs only differentiability.

The wall is the planner. "Compute $\bar V^*$" is value iteration, an iterative algorithm with a $\max$ and a sum inside, normally run as a black-box solver — how do I backpropagate into $f_R$ and $f_P$ through it? The resolution comes from writing value iteration out on the grid and looking at it:
$$Q_n(\bar s,\bar a)=\bar R(\bar s,\bar a)+\gamma\sum_{\bar s'}\bar P(\bar s'\mid\bar s,\bar a)V_n(\bar s'),\qquad V_{n+1}(\bar s)=\max_{\bar a}Q_n(\bar s,\bar a).$$
The state $\bar s$ is a cell $(i,j)$, so $V_n$ is a 2-D array — a single-channel image. Because transitions are local, $\bar P(\bar s'\mid\bar s,\bar a)$ is zero unless $\bar s'$ neighbors $\bar s$, so for a fixed action the sum $\sum_{\bar s'}\bar P(\bar s'\mid\bar s,\bar a)V_n(\bar s')$ takes, at each cell, the same weighted combination of $V_n$ over that cell's neighbors — that is a **convolution** of $V_n$ with a kernel whose entries are the (discounted) transition probabilities for that action. Adding the reward image $\bar R(\cdot,\bar a)$ gives the $Q$-image for $\bar a$. Stack one such convolution per action and the entire $Q$ tensor is *one convolution layer*: each output channel is $Q_n(\cdot,\bar a)$ and the kernel weights are the discounted transition probabilities. Then $V_{n+1}(\bar s)=\max_{\bar a}Q_n(\bar s,\bar a)$ is a **max over the action (channel) axis** — a channel-wise max that collapses the action channels back to a single value image. Crucially this is *not* the usual spatial max-pooling that downsamples; I must keep the value field at full resolution, so the max is over actions at a fixed cell. One VI sweep is thus a convolution producing per-action $Q$ channels followed by a channel-max producing the next value image; feeding that value image (stacked with the reward) back through the same conv-and-channel-max, $K$ times, performs $K$ sweeps of value iteration — and every operation is a convolution or a max, both differentiable. The planner becomes a recurrent convolutional module I can backpropagate the policy loss straight through into $\bar R$ and the transition kernels. In practice the first backup is computed from the reward alone, then each subsequent sweep convolves the stack $[\,\bar R,\, \bar V\,]$ with a kernel split into a reward-side part (the role of $\bar R$) and a value-feedback part (the role of $\gamma\bar P$, how last sweep's values propagate into this sweep's $Q$).

Several design choices follow from this and each earns its place. The kernels must be **tied across all $K$ sweeps**, because value iteration applies the *identical* Bellman operator every sweep; tying is both faithful and frugal — it gives the module effective computational depth $K$ at one sweep's worth of parameters, whereas an untied stack of $K$ conv layers has $K$ times the parameters, no force compelling each layer to implement the *same* operator, and so needs far more data and loses the very structure that makes this a planner. Untying degrades performance, hardest where data is scarce. The transition kernels are $3\times3$, because grid moves reach only immediate neighbors and locality is what keeps the parameter count tiny. The number of sweeps $K$ is set by information flow: each sweep is one convolution and so propagates value outward by one cell, meaning a state $L$ steps from the goal is goal-blind until at least $L$ sweeps have run; $K$ must therefore scale with the map diameter — about $10$ for $8\times8$, $20$ for $16\times16$, $36$ for $28\times28$ — kept as small as possible while still letting goal information reach every cell (too small and distant states guess; too large and the recurrent net is needlessly deep and costly). The $Q$ layer is given a few extra channels (e.g. $l_q=10$ for $8$ true actions), since the learned planner's latent action set need not match the real one — $\bar M$ is a caricature — and the extra freedom tends to help. The reward front-end $f_R$ is a small conv net: a $3\times3$ convolution to $l_h=150$ hidden channels, then a collapse to a single reward channel; it learns a reward-*shaping* — strongly negative at obstacles, strongly positive at the goal, slightly negative elsewhere — a useful fiction rather than the literal task reward, and the transitions $\bar P$ are just fixed-shape learned $3\times3$ kernels independent of the observation. The remarkable consequence is that the network is never told the true transition model yet, by shaping the reward end-to-end, learns to plan successful trajectories anyway. Finally the attention is trivial on a grid: the agent's real position maps one-to-one to a cell, so I simply index the $\bar Q$ tensor at the current cell $(i_s,j_s)$, pull out the vector of $Q$-values across action channels, and feed it to a linear readout and softmax.

Tracing it end to end against the failure I started from: a new map arrives, $f_R$ turns it into a reward image with negative walls and a positive goal, the VI module convolves-and-channel-maxes $K$ times to flood value out from the goal and route it around the walls, the attention reads the local $Q$ at my cell, and the readout climbs that field toward the goal. On a brand-new obstacle layout this simply works, because the network is *running* the planning computation rather than recalling a memorized output, and it holds up as the map grows so long as $K$ scales with it. This is more than "a deep enough CNN": a depth-$K$ CNN has the raw capacity, but capacity is not the issue — the VI module bakes in locality (small kernels), the repeated identical Bellman operator (tied weights), and channel-max action selection, and that inductive bias is what makes "learn to plan" learnable from a realistic amount of data. The 2-D grid is just the cleanest instance, where "local transitions $\Rightarrow$ convolution" is literal; for a graph (web pages and links) the same construction uses graph-structured transitions, a learned per-node reward from the query, and an attention reading values over relevant nodes, and for continuous control the attention selects a small patch of the value map around the discretized position with an FC head emitting continuous controls. The essential content is a differentiable, weight-tied, iterated Bellman backup.

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
