# Prioritized Experience Replay

## Problem

A value-based RL agent (DQN / Double DQN) stores transitions in a sliding-window replay memory and samples minibatches **uniformly at random**. Uniform sampling replays every transition at the frequency it was experienced, regardless of how much the agent can still learn from it, so most updates land on transitions the network already predicts well. In environments where informative transitions are rare and buried among redundant ones, this is very wasteful: a hindsight oracle that replays transitions in the most loss-reducing order is *exponentially* faster than uniform on a controlled needle-in-a-haystack task. Prioritized Experience Replay replaces uniform sampling with a cheap, online approximation of that oracle.

## Key idea

Replay transitions in proportion to how much can still be learned from them, proxied by the magnitude of the TD error `|δ|`, while (1) keeping sampling stochastic so no transition is starved and noise spikes can't dominate, and (2) correcting the bias that non-uniform sampling introduces into the learning target with importance-sampling weights.

## Method

**Priority.** Each transition `i` has priority `p_i > 0`. Two variants:
- Proportional: `p_i = |δ_i| + ε` (ε > 0 keeps zero-error transitions sampleable).
- Rank-based: `p_i = 1 / rank(i)`, rank by `|δ_i|` (robust to outliers/scale; power-law, heavy-tailed).

**Sampling distribution.** Sample transition `i` with
```
P(i) = p_i^α / Σ_k p_k^α.
```
`α = 0` recovers uniform; larger `α` is more greedy. Full support for any finite `α`.

**Bias and its correction.** Uniform replay makes SGD converge to the fixed point of `E_{i∼U}[δ_i ∇Q_i] = (1/N) Σ_i δ_i ∇Q_i = 0`. Sampling from `P` instead drives convergence to `E_{i∼P}[δ_i ∇Q_i] = Σ_i P(i) δ_i ∇Q_i = 0` — a different solution. Importance-sampling weights restore the uniform expectation:
```
w_i = ( (1/N) · 1/P(i) )^β = ( N · P(i) )^{-β},
```
since at `β = 1`, `E_{i∼P}[w_i δ_i ∇Q_i] = Σ_i P(i) · (1/(N P(i))) · δ_i ∇Q_i = (1/N) Σ_i δ_i ∇Q_i`. Use `w_i δ_i` in place of `δ_i` (weighted IS). Normalize by `1/max_j w_j` so weights only scale updates **downward** (stability). Anneal `β` linearly from `β_0` to `1`: leave the bias mostly uncorrected early (training is non-stationary anyway, and aggressive prioritization buys speed) and fully correct near convergence.

**Bookkeeping.** New transitions enter at maximal priority (seen at least once). After replaying transition `j`, set `p_j ← |δ_j|`. Cut the step-size `η` by ~4× vs. the uniform baseline, because prioritization raises typical gradient magnitudes. Built on Double DQN: target `y = R + γ Q_target(S', argmax_a Q(S', a))`.

**Efficient sampling.** A **sum-tree** (internal node = sum of children, leaves = `p_i^α`) gives `O(log N)` priority update and `O(log N)` sampling via prefix-sum descent. For a minibatch of `k`, split `[0, p_total]` into `k` equal segments and draw one sample per segment (stratified). A parallel **min-tree** supplies `min_k p_k^α` for the weight normalizer. (Rank-based instead uses an array heap as an approximately-sorted array, re-sorted rarely, with a precomputed `k`-segment power-law CDF.) Typical hyperparameters: rank-based `α=0.7, β_0=0.5`; proportional `α=0.6, β_0=0.4`.

### Algorithm (Double DQN with proportional prioritization)

```
Input: minibatch k, step-size η, replay period K, memory size N, exponents α, β, budget T
Initialize H = ∅, Δ = 0, p_1 = 1
Observe S_0, choose A_0 ∼ π_θ(S_0)
for t = 1 to T:
    observe S_t, R_t, γ_t
    store (S_{t-1}, A_{t-1}, R_t, γ_t, S_t) in H with priority p_t = max_{i<t} p_i
    if t ≡ 0 mod K:
        for j = 1 to k:
            sample j ∼ P(j) = p_j^α / Σ_i p_i^α
            w_j = (N · P(j))^{-β} / max_i w_i
            δ_j = R_j + γ_j Q_target(S_j, argmax_a Q(S_j, a)) − Q(S_{j-1}, A_{j-1})
            p_j ← |δ_j|
            Δ ← Δ + w_j · δ_j · ∇_θ Q(S_{j-1}, A_{j-1})
        θ ← θ + η · Δ ; Δ = 0
        from time to time: θ_target ← θ
    choose A_t ∼ π_θ(S_t)
```

## Code

Sum-tree / min-tree data structure:

```python
import operator, random
import numpy as np

class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2*node, node_start, mid)
        elif mid + 1 <= start:
            return self._reduce_helper(start, end, 2*node+1, mid+1, node_end)
        return self._operation(
            self._reduce_helper(start, mid, 2*node, node_start, mid),
            self._reduce_helper(mid+1, end, 2*node+1, mid+1, node_end))

    def reduce(self, start=0, end=None):
        if end is None: end = self._capacity
        if end < 0: end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2*idx], self._value[2*idx+1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operator.add, 0.0)
    def sum(self, start=0, end=None):
        return super().reduce(start, end)
    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:
            if self._value[2*idx] > prefixsum:
                idx = 2*idx
            else:
                prefixsum -= self._value[2*idx]
                idx = 2*idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, min, float('inf'))
    def min(self, start=0, end=None):
        return super().reduce(start, end)
```

Prioritized replay buffer (proportional variant):

```python
class PrioritizedReplayBuffer:
    def __init__(self, size, alpha):
        self._storage, self._maxsize, self._next_idx = [], size, 0
        self._alpha = alpha
        cap = 1
        while cap < size: cap *= 2
        self._it_sum = SumSegmentTree(cap)
        self._it_min = MinSegmentTree(cap)
        self._max_priority = 1.0

    def __len__(self): return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        idx = self._next_idx
        data = (obs_t, action, reward, obs_tp1, done)
        if idx >= len(self._storage): self._storage.append(data)
        else: self._storage[idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._it_sum[idx] = self._max_priority ** self._alpha   # new -> max priority
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        seg = p_total / batch_size
        for i in range(batch_size):                              # stratified
            mass = random.random() * seg + i * seg
            res.append(self._it_sum.find_prefixsum_idx(mass))
        return res

    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)     # (N * P_min)^(-beta)
        weights = []
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()    # P(i)
            w = (p_sample * len(self._storage)) ** (-beta)       # (N * P(i))^(-beta)
            weights.append(w / max_weight)                       # normalize, w <= 1
        return self._encode_sample(idxes), np.array(weights), idxes

    def update_priorities(self, idxes, priorities):
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def _encode_sample(self, idxes):
        obs, act, rew, obs2, done = [], [], [], [], []
        for i in idxes:
            o, a, r, o2, d = self._storage[i]
            obs.append(np.array(o, copy=False)); act.append(a); rew.append(r)
            obs2.append(np.array(o2, copy=False)); done.append(d)
        return (np.array(obs), np.array(act), np.array(rew),
                np.array(obs2), np.array(done))
```

Training step (Double-DQN target, IS-weighted update, priority write-back):

```python
def train_step(buffer, online_net, target_net, opt, batch_size, beta, gamma, eps=1e-6):
    (obs, act, rew, obs2, done), weights, idxes = buffer.sample(batch_size, beta)
    next_a = online_net(obs2).argmax(axis=1)
    y = rew + (1.0 - done) * gamma * target_net(obs2)[range(batch_size), next_a]
    delta = y - online_net(obs)[range(batch_size), act]
    loss = (weights * huber(delta)).mean()           # w_j * delta_j
    opt.zero_grad(); loss.backward(); opt.step()      # eta ~ baseline/4
    buffer.update_priorities(idxes, np.abs(delta) + eps)   # p_j = |delta_j| + eps
```
