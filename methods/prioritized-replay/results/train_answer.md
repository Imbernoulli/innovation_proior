A value-based agent — Q-learning with a deep network — learns not from each transition once but from a sliding-window buffer of the last million transitions, pulling a minibatch of 32 uniformly at random every few environment steps and taking a gradient step. That uniform draw was put there only to decorrelate consecutive transitions, and it does that job, but look at what it does to the learning itself: it replays every transition at exactly the frequency it happened to occur. A transition the network already predicts perfectly is pulled just as often as one the network is badly wrong about. This is wasteful, because the entire reason to have a network is that updating on a transition only helps when the network is wrong about it; once the temporal-difference error $\delta \approx 0$ for a transition, replaying it does essentially nothing. In environments where the informative transitions are rare and buried under a mass of redundant ones — a sparse final reward reached only after a long precise sequence of actions, surrounded by near-identical failures — uniform replay spends almost all of its updates on transitions it has already mastered, and environment interaction, not compute or memory, is the scarce resource.

How much does the order matter? Build the cleanest case where informative transitions are rare: a chain of $n$ states with two actions each, one "right" and one "wrong" that ends the episode immediately, where only the all-right sequence — probability $2^{-n}$ under random actions — reaches a reward of $1$, and the encoding blocks any generalization about which action is right. With all $2^{n+1}-2$ transitions sitting in the buffer, compare uniform replay against an oracle that at each step replays the single transition whose update, in hindsight, most reduces the total loss. The oracle is cheating — it needs hindsight over the whole buffer — but it bounds the ceiling, and the gap is enormous: against uniform, the oracle converges with a number of updates that looks exponentially smaller as $n$ grows, the two curves diverging as straight lines of different slope on a log-log plot. So order is not a ten-percent effect; in this regime it is the difference between learning and not learning. The remaining question is purely how to approximate that oracle cheaply, online, and model-free.

I propose Prioritized Experience Replay. The oracle wants the transition whose update reduces loss the most, and the Q-learning update already moves the parameters by $\eta\,\delta\,\nabla_\theta Q$ with $\delta = R + \gamma \max_a Q(S',a) - Q(S,A)$, so a transition with $\delta \approx 0$ produces a near-zero update while one with large $|\delta|$ is where the current estimate is far from its own bootstrap target — the network is surprised, and the step moves things a lot. Thus $|\delta|$ is a proxy for how much a transition can still teach, and it is already computed on every update, so it costs nothing. The crudest use of it is greedy: store each transition's last-seen $|\delta|$, always replay the largest, and write back the fresh $\delta$ the update yields; new transitions get maximal priority so each is seen at least once, with a binary heap giving $O(1)$ max-finding and $O(\log N)$ replacement. But greedy is brittle in three distinct ways. Priorities of unreplayed transitions go stale — a transition with a small $|\delta|$ the one time it was seen is parked low forever and, under a sliding window, drops out before it is reconsidered even if it later became informative. Noise corrupts it — stochastic rewards or, worse, bootstrap approximation error can inflate $|\delta|$ purely by chance, and greedy locks onto those spikes. And under function approximation errors shrink slowly, so the same tiny high-error subset is replayed over and over, collapsing diversity and overfitting a sliver of the buffer. A pure argmax over $|\delta|$ is simply too sharp. (Greedy also has a degenerate trap: if all $Q$-values start at zero, every unrewarded transition has target $0+\gamma\cdot 0 = 0$ and estimate $0$, so $\delta = 0$ exactly, all sink to the heap bottom, and nothing is revisited — it needs randomized or optimistic initialization.)

What makes the method work is to soften greedy without retreating to uniform, with a single knob that slides between them. Make the probability of sampling a transition monotone in its priority but strictly positive for every transition, so nothing is permanently abandoned, a lone noise spike only inflates a probability rather than seizing all the mass, and diversity is preserved. The natural one-parameter family takes priority $p_i > 0$, raises it to a power $\alpha$, and normalizes,

$$P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}.$$

At $\alpha = 0$ every $p_i^0 = 1$ and $P(i) = 1/N$, recovering uniform exactly as a special case; as $\alpha \to \infty$ the largest priority dominates and it approaches greedy; in between, $\alpha$ dials the aggressiveness with full support guaranteed for any finite $\alpha$. For the priority itself there are two choices. Proportional uses $p_i = |\delta_i| + \varepsilon$, where the small $\varepsilon$ is load-bearing: without it a transition whose error hits exactly zero would get $p_i = 0$ and leave the support, the very failure being prevented. Rank-based discards magnitudes and uses $p_i = 1/\mathrm{rank}(i)$, the position when the buffer is sorted by $|\delta|$; throwing away magnitude buys robustness, because rank knows only that one error exceeds another, not by how much, so a wildly outlying $|\delta|$ — a noise spike or a bootstrap blow-up — cannot grab a disproportionate slice but just sits at rank one, and $1/\mathrm{rank}$ under the $\alpha$ power becomes a heavy-tailed power law that mechanically guarantees diversity. The cost is blindness to error scale, so where the magnitudes carry genuine exploitable structure — one transition's error truly dwarfing the rest — proportional wins because it can see that. Both are kept.

The detail that cannot be skipped is that changing the sampling distribution changes the destination, not just the path to it. Uniform replay, sampling $i \sim \mathrm{Uniform}$ and applying $\theta \leftarrow \theta + \eta\,\delta_i \nabla_\theta Q_i$, drives SGD toward parameters where the expected update vanishes,

$$\mathbb{E}_{i\sim U}\!\left[\delta_i \nabla_\theta Q_i\right] = \frac{1}{N}\sum_i \delta_i \nabla_\theta Q_i = 0,$$

and that $1/N$ weighting *is* the objective whose fixed point defines the solution. Sampling $i \sim P$ instead and applying the same $\delta_i \nabla_\theta Q_i$ drives SGD to where $\sum_i P(i)\,\delta_i \nabla_\theta Q_i = 0$, a different weighted problem with a different solution — the prioritization is biased in the destination. Importance sampling fixes this: I want expectations under the target $1/N$ but I am drawing from $P$, so I multiply each sampled term by the density ratio target-over-sampling,

$$w_i = \frac{1/N}{P(i)},$$

and then under $P$,

$$\mathbb{E}_{i\sim P}\!\left[w_i\,\delta_i \nabla_\theta Q_i\right] = \sum_i P(i)\cdot\frac{1/N}{P(i)}\cdot \delta_i \nabla_\theta Q_i = \frac{1}{N}\sum_i \delta_i \nabla_\theta Q_i = \mathbb{E}_{i\sim U}\!\left[\delta_i \nabla_\theta Q_i\right],$$

so $P(i)$ cancels exactly and the followed gradient is the unbiased one — draw from $P$ for the speed, use $w_i\delta_i$ wherever $\delta_i$ appeared. The magnitudes bite, though: a high-priority transition has large $P(i)$, so its $w_i$ is small, scaling down exactly what is oversampled, which is the correction; but a low-priority transition has tiny $P(i)$, so $w_i$ blows up and a rare draw could throw a wild step. So normalize by the maximum weight over the replay memory, $w_i \leftarrow w_i / \max_l w_l$, making the largest weight $1$ and every update only ever scaled downward. Since $\max_l w_l$ is a single scalar for the current memory, at $\beta = 1$ this scales the whole expected update without moving its zero, bounding step size while preserving the corrected fixed point.

I do not, however, want the full correction throughout. The unbiasedness argument is about the converged fixed point — it matters at the end of training. Early on the whole process is wildly non-stationary: the policy changes, the induced state distribution drifts, and the bootstrap target $\theta_{\text{target}}$ shifts; insisting on an exactly-unbiased gradient with respect to a moving target is correcting for a precision that does not exist, and a small early bias is in the noise. Worse, the full correction actively fights the prioritization, scaling down precisely the high-priority transitions whose extra weight buys the early speedup. So I want little correction early and full correction late, via a second exponent on the weight,

$$w_i = \left(\frac{1}{N}\cdot\frac{1}{P(i)}\right)^{\beta} = \left(N\cdot P(i)\right)^{-\beta},$$

which is $w_i = 1$ (no correction, pure prioritized speed) at $\beta = 0$ and the full IS weight at $\beta = 1$. I anneal $\beta$ linearly from a starting $\beta_0 < 1$ up to exactly $1$ at the end of learning. The exponents are a push-pull pair: cranking $\alpha$ prioritizes harder, cranking $\beta$ corrects harder, and one retreats toward the baseline by lowering $\alpha$ or raising $\beta$. There is also a free stabilizer here. With non-linear function approximation the gradient is only a local first-order picture; a big step walks off where the linearization is valid. But prioritization guarantees a high-error transition is seen many times while the normalized IS weight ($\le 1$) shrinks each individual step, so instead of one big leap I take many small re-linearized steps along the same direction, following curvature instead of overshooting it; and as $\beta \to 1$ the normalizer $\max_l w_l$ grows, steadily shrinking the average effective step, so annealing doubles as an implicit step-size decay.

The implementation must be honest about cost, because at $N = 10^6$ nothing linear in $N$ per sample is affordable. For the proportional variant, store the $p_i^{\alpha}$ values at the leaves of a binary tree where each internal node holds the sum of its children, so the root holds $\sum_k p_k^{\alpha}$; updating one leaf and walking to the root is $O(\log N)$. To draw a sample, pick a target mass $u \in [0, p_{\text{total}}]$ and descend from the root — at each node go left if the left child's sum exceeds $u$, else subtract that sum and go right — landing at the leaf whose cumulative interval contains $u$, a prefix-sum search that is $O(\log N)$ and hits leaf $i$ with probability exactly $P(i)$. For a minibatch of $k$, split $[0, p_{\text{total}}]$ into $k$ equal-mass segments and draw one uniform $u$ from each; this stratification lowers the minibatch variance. To get the normalizer I need $\max_l w_l$, which since $w$ decreases in $P$ comes from the smallest $P(i)$, i.e. the smallest $p_i^{\alpha}$, so a parallel min-tree over the same leaves supplies $\min_l p_l^{\alpha}$ in $O(\log N)$. (The rank-based variant instead keeps an array heap treated as an approximately-sorted array, re-sorted only rarely, with a precomputed $k$-segment power-law CDF.) Everything else is Double DQN with exactly the replay changed: build the target $y = R + \gamma\, Q_{\text{target}}(S', \arg\max_a Q(S',a))$ to keep overestimation down, sample from $P$, fold $w_i$ into the update, write the fresh $|\delta_i| + \varepsilon$ back as the new score, and insert every new transition at the current maximum score so it is replayed at least once. One hyperparameter must move: prioritization picks high-$|\delta|$ transitions far more often than uniform, so the typical per-minibatch gradient magnitude rises, and keeping the old step-size would make the effective learning rate too hot, so $\eta$ is cut by a factor of $4$. Beyond that, $\alpha$ and $\beta_0$ are the only genuinely new knobs, with a coarse search landing near $\alpha = 0.7,\ \beta_0 = 0.5$ for rank-based and $\alpha = 0.6,\ \beta_0 = 0.4$ for proportional. One reassurance about the two variants: the base agent already clips rewards and TD errors to $[-1,1]$, which is itself an outlier-removal step that caps the magnitudes that would make proportional misbehave, so the variants should not separate merely on rare huge errors.

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
