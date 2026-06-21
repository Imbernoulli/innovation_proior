Noisy nets cleared the bar by a hair — 118% against 117% — and the flatness is the message: exploration was not what capped the median game, since the hard-exploration titles where parameter noise shines are a minority and the typical game already saw enough reward signal under $\epsilon$-greedy. So I want an axis that touches *every* game's learning, and one has been sitting in plain sight: how the replay buffer hands transitions to the update. The floor draws minibatches *uniformly at random*, a choice I only ever made to decorrelate samples — but decorrelation requires not training on the live stream, not uniformity, so uniform sampling is free to be wrong. And it is wrong: it replays each transition at the frequency it happened to be experienced, regardless of how much the agent can still learn from it. Most of the buffer, most of the time, is transitions the network already predicts well, with tiny TD error and so a tiny gradient, while the surprising transitions that would teach the most are sampled at exactly the same rate. On a game where informative transitions are rare and buried among redundant ones, almost the entire gradient budget is spent on transitions there is nothing left to learn from. The size of the prize is concrete: on a controlled needle-in-a-haystack task, an oracle that replays transitions in the order that most reduces the loss learns *exponentially* faster than uniform. I cannot build that oracle — it needs the future loss reductions — but I can approximate it with a quantity I already compute.

I propose **Prioritized Experience Replay**: sample transitions in proportion to how surprising they are, proxied by the TD-error magnitude $|\delta|$. A large $|\delta|$ means the current network's prediction is far from its own bootstrap target — surprising under the current value function, so a gradient step on it moves the weights a lot — and a small $|\delta|$ means the transition is already consistent with what the net believes; and I compute $\delta$ for every sampled transition anyway. So I give transition $i$ a priority $p_i=|\delta_i|+\varepsilon$ (the small $\varepsilon>0$ keeps a zero-error transition from becoming unsamplable) and sample it with probability $P(i)=p_i^\alpha/\sum_k p_k^\alpha$. The exponent $\alpha$ interpolates: $\alpha=0$ is back to uniform, larger $\alpha$ is greedier toward high-error transitions. I keep it *stochastic* rather than always taking the single highest-error transition for two reasons — a greedy argmax would replay a small set over and over and starve the rest of the buffer (a transition whose error happened to start low would never be revisited and corrected), and TD errors are noisy, so a one-off error spike should not let a transition dominate. Stochastic prioritization with full support fixes both, since every transition keeps a non-zero probability.

But prioritized sampling introduces a bias I have to face, because it changes *which fixed point* the updates converge to. Uniform SGD drives the weights to where $\mathbb{E}_{i\sim U}[\delta_i\nabla Q_i]=\frac1N\sum_i\delta_i\nabla Q_i=0$ — the gradient averaged over the *empirical data distribution*. Sampling from $P$ instead drives convergence to $\mathbb{E}_{i\sim P}[\delta_i\nabla Q_i]=\sum_i P(i)\,\delta_i\nabla Q_i=0$, a *different* solution that overweights the high-priority transitions, while the value I actually want is still the one defined over the real data distribution. The correction is importance sampling: weight each sampled transition's update by $w_i=\big(\frac1N\cdot\frac1{P(i)}\big)^\beta=(N\,P(i))^{-\beta}$, the ratio of the target (uniform) probability to the sampling probability raised to a $\beta$ that controls how much bias I correct. At $\beta=1$ the correction is exact: $\mathbb{E}_{i\sim P}[w_i\delta_i\nabla Q_i]=\sum_i P(i)\frac{1}{N P(i)}\delta_i\nabla Q_i=\frac1N\sum_i\delta_i\nabla Q_i$ recovers the uniform expectation. I use $w_i\delta_i$ in place of $\delta_i$ in the update and normalize the weights by $1/\max_l w_l$ over the buffer so they only ever scale steps *downward* — pure stability, never blowing a step up. And I anneal $\beta$ from a $\beta_0<1$ up to $1$ over training: early on the network changes fast and the notion of a fixed point is moot, so I want the raw speed of aggressive prioritization; near the end, where the bias would actually pin the wrong solution, I correct it fully.

A few details make it compose cleanly. I build it on the decoupled target from rung 2 — the prioritized error is the Double-DQN error $\delta=R+\gamma\,Q_{\theta^-}(S',\arg\max_a Q_\theta(S',a))-Q_\theta(S,A)$ — because prioritizing a biased error would just prioritize the bias. A brand-new transition enters at *maximal* priority so it is guaranteed at least one replay (I have no error estimate for it yet); after I replay transition $i$ and compute its fresh $\delta_i$, I write the new priority $p_i\leftarrow|\delta_i|+\varepsilon$ back into the buffer. And because prioritization raises the *typical* gradient magnitude (I am deliberately sampling the big-error transitions), I cut the learning rate by roughly $4\times$ versus uniform so the effective step size stays in range. The one implementation worry is cost: naively sampling from $P(i)\propto p_i^\alpha$ and maintaining $\sum_k p_k^\alpha$ over a $10^6$-entry buffer is $O(N)$ per draw — far too slow. A **sum-tree** fixes it: a binary tree whose leaves hold $p_i^\alpha$ and whose internal nodes hold the sum of their children, so updating one leaf is $O(\log N)$ (walk up fixing sums) and sampling is $O(\log N)$ via prefix-sum descent — draw a uniform value in $[0,\text{total}]$ and walk down, going left or right by comparing against the left child's sum. For a minibatch of $k$ I stratify, splitting $[0,\text{total}]$ into $k$ equal segments and drawing one sample per segment so the batch spreads across the priority range. A parallel **min-tree** gives $\min_i p_i^\alpha$ in $O(\log N)$ for the weight normalizer $\max_l w_l$. The whole scheme is $O(\log N)$ per transition — negligible against a conv forward/backward.

The bar: unlike noisy nets, this touches the data efficiency of *every* game, since every game has redundant transitions it over-replays and rare informative ones it under-replays, so reallocating the gradient budget toward what is still learnable should lift the broad middle of the distribution rather than the tails — exactly what moves a median. The risk is the bias: if the importance-sampling correction is mis-set the agent could converge to a subtly wrong value function and lose on games where the fixed point matters, so the $\beta$-anneal is doing real work. But if the analysis is right, this is the first axis since the decoupled target that should give a *broad* lift — I expect a clear jump above 118%, the largest single step since 117%.

```python
# Prioritized replay: sum-tree/min-tree buffer + IS-weighted Double-DQN update + priority write-back.
# Code home: vwxyzjn/cleanrl + dopamine; excerpted from methods/prioritized-replay/results/answer.md.
import operator, random
import numpy as np
import torch
import torch.nn.functional as F


def huber(delta, kappa=1.0):
    return F.huber_loss(delta, torch.zeros_like(delta), delta=kappa, reduction="none")


class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operator.add, 0.0)

    def sum(self):
        return self._value[1]

    def find_prefixsum_idx(self, prefixsum):
        idx = 1
        while idx < self._capacity:                       # prefix-sum descent: O(log N) sampling
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, min, float("inf"))

    def min(self):
        return self._value[1]


class PrioritizedBuffer:
    def __init__(self, size, alpha):
        self._storage, self._maxsize, self._next = [], size, 0
        self._alpha, self._max_p = alpha, 1.0
        cap = 1
        while cap < size:
            cap *= 2
        self._sum, self._min = SumSegmentTree(cap), MinSegmentTree(cap)

    def add(self, transition):
        idx = self._next
        if idx >= len(self._storage):
            self._storage.append(transition)
        else:
            self._storage[idx] = transition
        self._next = (self._next + 1) % self._maxsize
        self._sum[idx] = self._min[idx] = self._max_p ** self._alpha   # new -> max priority

    def sample(self, batch_size, beta):
        n, total = len(self._storage), self._sum.sum()
        seg = total / batch_size
        idxes = [self._sum.find_prefixsum_idx(random.random() * seg + i * seg)  # stratified
                 for i in range(batch_size)]
        p_min = self._min.min() / total
        max_w = (p_min * n) ** (-beta)
        weights = np.array([((self._sum[i] / total) * n) ** (-beta) / max_w for i in idxes],
                           dtype=np.float32)                                   # normalized, w <= 1
        return [self._storage[i] for i in idxes], weights, idxes

    def update_priorities(self, idxes, deltas, eps=1e-6):
        for i, d in zip(idxes, deltas):
            p = abs(d) + eps                                                   # p_i = |delta_i| + eps
            self._sum[i] = self._min[i] = p ** self._alpha
            self._max_p = max(self._max_p, p)


def train_step(buffer, online_net, target_net, opt, batch_size, beta, gamma):
    (obs, act, rew, next_obs, done), weights, idxes = buffer.sample(batch_size, beta)
    next_a = online_net(next_obs).argmax(1)                                    # Double-DQN target
    y = rew + (1.0 - done) * gamma * target_net(next_obs)[range(len(next_obs)), next_a]
    delta = y - online_net(obs)[range(len(obs)), act]
    loss = (torch.as_tensor(weights) * huber(delta)).mean()                   # IS weight on the gradient
    opt.zero_grad(); loss.backward(); opt.step()                              # lr ~ baseline / 4
    buffer.update_priorities(idxes, delta.detach().cpu().numpy())             # priority write-back
```
