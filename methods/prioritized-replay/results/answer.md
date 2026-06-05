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
since at `β = 1`, `E_{i∼P}[w_i δ_i ∇Q_i] = Σ_i P(i) · (1/(N P(i))) · δ_i ∇Q_i = (1/N) Σ_i δ_i ∇Q_i`. Use `w_i δ_i` in place of `δ_i` (weighted IS). Normalize by `1/max_l w_l` over the replay memory so weights only scale updates **downward** (stability). Anneal `β` linearly from `β_0` to `1`: leave the bias mostly uncorrected early (training is non-stationary anyway, and aggressive prioritization buys speed) and fully correct near convergence.

**Bookkeeping.** New transitions enter at maximal priority (seen at least once). For the proportional variant, after replaying transition `i`, set `p_i ← |δ_i| + ε`. Cut the step-size `η` by ~4× vs. the uniform baseline, because prioritization raises typical gradient magnitudes. Built on Double DQN: target `y = R + γ Q_target(S', argmax_a Q(S', a))`.

**Efficient sampling.** A **sum-tree** (internal node = sum of children, leaves = `p_i^α`) gives `O(log N)` priority update and `O(log N)` sampling via prefix-sum descent. For a minibatch of `k`, split `[0, p_total]` into `k` equal segments and draw one sample per segment (stratified). A parallel **min-tree** supplies `min_i p_i^α` for the weight normalizer. (Rank-based instead uses an array heap as an approximately-sorted array, re-sorted rarely, with a precomputed `k`-segment power-law CDF.) Typical hyperparameters: rank-based `α=0.7, β_0=0.5`; proportional `α=0.6, β_0=0.4`.

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
            sample index i_j ∼ P(i) = p_i^α / Σ_l p_l^α
            w_{i_j} = (N · P(i_j))^{-β} / max_l w_l
            δ_{i_j} = R_{i_j} + γ_{i_j} Q_target(S_{i_j}, argmax_a Q(S_{i_j}, a)) − Q(S_{i_j-1}, A_{i_j-1})
            p_{i_j} ← |δ_{i_j}| + ε
            Δ ← Δ + w_{i_j} · δ_{i_j} · ∇_θ Q(S_{i_j-1}, A_{i_j-1})
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

Replay buffer with score-based proportional sampling:

```python
class ReplayBuffer:
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self): return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage): self._storage.append(data)
        else: self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obs, act, rew, obs2, done = [], [], [], [], []
        for i in idxes:
            o, a, r, o2, d = self._storage[i]
            obs.append(np.asarray(o)); act.append(a); rew.append(r)
            obs2.append(np.asarray(o2)); done.append(d)
        return (np.array(obs), np.array(act), np.array(rew),
                np.array(obs2), np.array(done))

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)


class WeightedReplayBuffer(ReplayBuffer):
    def __init__(self, size, score_exponent):
        super().__init__(size)
        assert score_exponent >= 0
        self._score_exponent = score_exponent
        cap = 1
        while cap < size: cap *= 2
        self._it_sum = SumSegmentTree(cap)
        self._it_min = MinSegmentTree(cap)
        self._max_score = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        weighted_score = self._max_score ** self._score_exponent
        self._it_sum[idx] = weighted_score     # new -> max score
        self._it_min[idx] = weighted_score

    def _sample_weighted(self, batch_size):
        res = []
        total = self._it_sum.sum(0, len(self._storage))
        seg = total / batch_size
        for i in range(batch_size):                              # stratified
            mass = random.random() * seg + i * seg
            res.append(self._it_sum.find_prefixsum_idx(mass))
        return res

    def sample(self, batch_size, correction_exponent):
        idxes = self._sample_weighted(batch_size)
        total = self._it_sum.sum(0, len(self._storage))
        p_min = self._it_min.min(0, len(self._storage)) / total
        max_weight = (p_min * len(self._storage)) ** (-correction_exponent)
        weights = []
        for idx in idxes:
            p_sample = self._it_sum[idx] / total                 # P(i)
            w = (p_sample * len(self._storage)) ** (-correction_exponent)
            weights.append(w / max_weight)                       # normalize, w <= 1
        return self._encode_sample(idxes), np.array(weights, dtype=np.float32), idxes

    def update_scores(self, idxes, scores):
        assert len(idxes) == len(scores)
        for idx, score in zip(idxes, scores):
            assert score > 0
            assert 0 <= idx < len(self._storage)
            weighted_score = score ** self._score_exponent
            self._it_sum[idx] = weighted_score
            self._it_min[idx] = weighted_score
            self._max_score = max(self._max_score, score)
```

Training step (Double-DQN target, IS-weighted update, priority write-back):

```python
def td_error(batch, online_net, target_net, gamma):
    s, a, r, s2, done = batch
    next_a = online_net(s2).argmax(axis=1)
    y = r + (1.0 - done) * gamma * target_net(s2)[range(len(s2)), next_a]
    return y - online_net(s)[range(len(s)), a]


def train_step(buffer, online_net, target_net, opt, batch_size,
               correction_exponent, gamma, score_floor=1e-6):
    batch, weights, idxes = buffer.sample(batch_size, correction_exponent)
    delta = td_error(batch, online_net, target_net, gamma)
    loss = (weights * huber(delta)).mean()           # gradient gets the w_i factor
    opt.zero_grad(); loss.backward(); opt.step()      # eta ~ baseline/4
    buffer.update_scores(idxes, np.abs(delta) + score_floor)
```
