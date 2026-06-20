# Prioritized Experience Replay — sample by TD error

**Problem.** Uniform replay replays every transition at the frequency it was experienced, regardless of how
much is left to learn from it — so the gradient budget is mostly spent on transitions the network already
predicts well. On games where informative transitions are rare and buried among redundant ones this is very
wasteful (a loss-greedy oracle replay is *exponentially* faster than uniform on a needle-in-a-haystack
task). This wastes budget on *every* game, not a minority.

**Key idea.** Replay transitions in proportion to how surprising they are, proxied by the TD-error
magnitude $|\delta|$ (a quantity already computed for the loss): priority $p_i=|\delta_i|+\varepsilon$,
sample $P(i)=p_i^\alpha/\sum_k p_k^\alpha$ ($\alpha=0$ recovers uniform). Kept *stochastic* with full
support so no transition is starved and noise spikes can't dominate.

**Bias correction.** Non-uniform sampling moves the SGD fixed point, so reweight with importance-sampling
weights $w_i=(N\,P(i))^{-\beta}$ (exact at $\beta=1$), normalized by $1/\max_l w_l$ so steps only scale
*down* (stability); anneal $\beta:\beta_0\!\to\!1$ (bias matters only near convergence). Built on the
Double-DQN target. New transitions enter at max priority; write back $p_i\!\leftarrow\!|\delta_i|+\varepsilon$
after replay; cut the learning rate $\sim4\times$ (prioritization raises typical gradient magnitude).

**Efficiency.** A **sum-tree** (leaves $p_i^\alpha$, internal nodes = child sums) gives $O(\log N)$ update
and $O(\log N)$ stratified sampling via prefix-sum descent; a parallel **min-tree** supplies the weight
normalizer.

**Bar.** Unlike noisy nets, this touches the data efficiency of *every* game, so it should lift the broad
middle of the distribution — the first broad gain since the decoupled target.

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
