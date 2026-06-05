OK, let me think about replay from scratch. I have a value-based agent — Q-learning with a deep network — and it learns from a sliding-window buffer of the last million transitions. Every few steps I pull 32 of them uniformly at random and take a gradient step. That uniform draw was put there to decorrelate consecutive transitions and it does that job, but stare at what it actually does to the *learning*: it replays every transition at exactly the frequency it happened to occur. A transition the network already predicts perfectly gets pulled just as often as one the network is badly wrong about. That feels wasteful. The whole point of having a network is that updating on a transition only helps if the network is wrong about it; once `δ ≈ 0` for a transition, replaying it does essentially nothing.

How wasteful, though? Let me build the cleanest case I can where the informative transitions are rare and see how much order matters. Picture a chain of `n` states, two actions at each: one is "right", one is "wrong" and ends the episode immediately. Only the all-right sequence — probability `2^{-n}` under random actions — reaches a reward of 1 at the very end; reward is zero everywhere else, and I'll set up the state encoding so the agent can't generalize about which action is "right". So the buffer is full of near-identical zero-reward failures and exactly one rewarding trajectory. This is the needle-in-a-haystack structure I care about, distilled.

Now suppose I have all `2^{n+1}-2` transitions sitting in the buffer and I just keep doing Q-learning updates until the value estimates converge. Two replay strategies. First, uniform. Second, an oracle: at each step it looks over the whole buffer and replays the single transition that, in hindsight after the update, most reduces the total loss. The oracle is cheating — it requires hindsight over the whole buffer — but it tells me the ceiling. And the gap is enormous: the oracle converges with a number of updates that, against uniform, looks exponentially smaller as `n` grows. On a log-log plot the two curves diverge as straight lines of different slope. So order isn't a 10% effect; in the regime I care about it's the difference between learning and not learning. That settles whether this is worth doing. The question is purely: what cheap, online, model-free proxy approximates that oracle?

What does the oracle actually want? The transition whose update reduces loss the most. The loss is built out of TD errors. The Q-learning update already moves the parameters by `η · δ · ∇_θ Q`, proportional to `δ = R + γ max_a Q(S',a) − Q(S,A)`. So a transition with `δ ≈ 0` produces a near-zero update — replaying it changes almost nothing. A transition with large `|δ|` is one where the current estimate is far from its own bootstrap target: the network is surprised there, and the update will move things a lot. `|δ|` is a proxy for "how much can I still learn from this transition", and crucially it's already computed every time I do an update — it costs me nothing extra. That's my proxy. Prioritize by `|δ|`.

The crudest thing that uses it: greedy. Store the last-seen `|δ|` alongside each transition; always replay the one with the largest `|δ|`; do the Q-update, which both moves the weights and yields a fresh `δ` for that transition, which I write back. New transitions I've never replayed have no `δ` yet — give them maximal priority so every transition is at least seen once. To do this at scale I keep a binary heap keyed on `|δ|`: finding the max is `O(1)`, replacing a priority after the update is `O(log N)`. This is the closest cheap model-free shape to the oracle: focus each update where the current Bellman residual says the stored value is most wrong.

(One trap I have to be careful about with greedy: if I initialize all Q-values to zero, then every unrewarded transition has target `0 + γ·0 = 0` and estimate `0`, so `δ = 0` exactly. They all look maximally uninformative, sink to the bottom of the heap, and never get revisited until some other transition's error drops below numerical precision — which could be never. So greedy needs a randomized or optimistic initialization to break that symmetry. Note that.)

But greedy is brittle, and I can see three distinct ways it breaks. First: I only refresh `|δ|` for transitions I actually replay. A transition that happened to have small `|δ|` the one time I saw it gets parked low in the heap forever — and with a sliding window it'll fall out before I ever reconsider it, even if it became informative later as the network changed. Its priority is permanently stale. Second: `|δ|` is noisy. If rewards are stochastic, or — more insidiously — if bootstrapping injects approximation error into the target, then a transition can show a large `|δ|` purely from noise, and greedy will lock onto exactly those noise spikes and chase them. Third, and this is the one that worries me most under function approximation: errors shrink *slowly*. The handful of transitions that start with high error keep high error for many updates, so greedy replays the same tiny subset over and over. That's a collapse of diversity — I'm overfitting to a sliver of the buffer and starving everything else. A pure argmax over `|δ|` is too sharp.

So I don't want pure greedy, but I obviously don't want pure uniform either — uniform is the thing I'm trying to beat. I want a knob that slides between them. Make the probability of sampling a transition *monotone* in its priority, but keep it strictly positive for every transition so nothing gets permanently abandoned (kills problem one and softens problem three) and so a single noise spike only inflates a probability rather than seizing all the mass (softens problem two). The natural one-parameter family that does this: take priority `p_i > 0`, raise it to a power `α`, and normalize,

```
P(i) = p_i^α / Σ_k p_k^α.
```

Look at the endpoints. At `α = 0` every `p_i^0 = 1`, so `P(i) = 1/N` — exactly uniform, my baseline falls out as a special case, good. As `α → ∞` the largest priority dominates and it approaches the argmax — greedy. In between, `α` dials exactly the aggressiveness I wanted, with full support guaranteed for any finite `α` because each `p_i > 0`. That's the interpolation.

Now what do I plug in for `p_i`? The direct choice is proportional: `p_i = |δ_i| + ε`. The `ε` is small but it's load-bearing — without it a transition whose error hits exactly zero would get `p_i = 0` and drop out of the support entirely, which is the very failure I'm trying to prevent; `ε` guarantees it can still be revisited. The other choice is to throw away the magnitudes and use only the ordering: `p_i = 1/rank(i)`, where `rank(i)` is the transition's position when the buffer is sorted by `|δ|`. Why would I want to discard magnitude information? Because of greedy's problem two — noise spikes and outliers. Rank only knows that one transition has larger error than another, not by how much, so a wildly outlying `|δ|` (a noise spike, or a stray bootstrap blow-up) can't grab a disproportionate slice of probability; it just sits at rank 1. And `1/rank` under the `α` power becomes a power-law distribution — heavy-tailed by construction, which mechanically guarantees the diversity I was worried about losing. The cost is that rank is blind to the actual scale of errors, so where there's real exploitable structure in the error magnitudes — say a sparse-reward problem where one transition's error genuinely dwarfs the rest — proportional should win because it can see that. So: proportional sees magnitude and exploits structure; rank is robust to outliers and scale and guarantees a heavy tail. I'll keep both.

I should not pretend `|δ|` is the only possible score. It is the simplest proxy for expected learning progress, but it can confuse "large and learnable" with "large because this transition is noisy, partially observed, or beyond the function approximator's capacity." One fix would be to score the *drop* in `|δ|` since the transition was last replayed: if a transition stays high no matter how often I train on it, maybe it is not actually teaching me. That is appealing, and an RProp-like history could even increase the score while consecutive signs agree and reduce it when signs flip, but now the score depends on everything replayed in between the two measurements, so the variance goes up and the bookkeeping is no longer just the TD error I already had. Another fix is to score the norm of the parameter change caused by the transition; that lets an adaptive optimizer suppress unlearnable noisy directions, but it delegates the core decision to the optimizer and costs more than reading off `|δ|`. I could also break the symmetry between positive and negative TD errors, because "better than expected" may be more informative than one of many ways of being worse than expected, or I could boost whole episodes by return-to-go. Those are plausible variants, but the low-cost default is still `|δ|`: one scalar, already computed, aligned with the actual update.

There are also alternatives to stochastic sampling for the diversity problem. I could add a novelty term in observation space, mix several priority rules inside each minibatch, or add a staleness bonus so a transition's chance of replay rises the longer its last TD error has gone unchecked. If that bonus grows linearly with time, I can implement it without incrementing every stored item by writing updated scores with a negative term proportional to the current global step; untouched scores then become relatively larger as time passes. With bootstrapping there is one more structural trick: when a transition produces a large TD error, it changes my estimate of its next state, so the transition immediately preceding it in the historical trajectory may now have a changed bootstrap target. Adding some of the current `|δ|` to that predecessor, when the predecessor is not terminal, is a model-free shadow of prioritized sweeping's backward propagation. I do not need any of these to define the clean core, but they clarify what the core is leaving on the table.

Now the thing that's been nagging me since I wrote `P(i) ≠ 1/N`. I changed the sampling distribution. Does that change *what the agent converges to*, or only how fast it gets there? Let me be careful, because if it changes the destination then "faster" is worthless.

What does the uniform-replay update converge to? Each step I sample `i ∼ Uniform`, take `θ ← θ + η · δ_i · ∇_θ Q_i`, and SGD drives me toward parameters where the *expected* update vanishes:

```
E_{i∼U}[ δ_i ∇_θ Q_i ] = (1/N) Σ_i δ_i ∇_θ Q_i = 0.
```

That averaging — each transition weighted `1/N` — *is* the objective. The fixed point is defined by it. Now if instead I sample `i ∼ P` and apply the same `δ_i ∇_θ Q_i`, SGD drives me to where `E_{i∼P}[δ_i ∇_θ Q_i] = Σ_i P(i) δ_i ∇_θ Q_i = 0`, which weights each transition by `P(i)` instead of `1/N`. That's a *different* set of equations with a *different* solution. So yes — prioritized sampling changes the fixed point itself, not just the path to it. It's biased, and the bias is in the destination. Even with the policy and the state distribution held fixed, I'd be solving the wrong weighted problem. I can't just ship the speedup and ignore this.

I need to keep the speed benefit of sampling from `P` while making the *expected update* still be the uniform one. This is exactly what importance sampling is for: I'm taking expectations under `P` but I want them under the target distribution `1/N`, so I multiply each sampled term by the ratio of the densities, target-over-sampling:

```
w_i = (1/N) / P(i).
```

Check that it works. Under `P`,

```
E_{i∼P}[ w_i δ_i ∇_θ Q_i ] = Σ_i P(i) · (1/N)/P(i) · δ_i ∇_θ Q_i = (1/N) Σ_i δ_i ∇_θ Q_i = E_{i∼U}[ δ_i ∇_θ Q_i ].
```

The `P(i)` cancels exactly and I'm back to the uniform expectation. So I draw from `P` for the speed, but I reweight by `w_i` and the gradient I follow in expectation is the unbiased one. Concretely I just use `w_i δ_i` wherever I used `δ_i`. The fixed point is restored to the right one.

Now think about magnitudes, because this is going to bite. A high-priority transition has large `P(i)`, so `w_i = (1/N)/P(i)` is *small* — exactly the transitions I'm oversampling get their contribution scaled down, which is the whole correction. But a low-priority transition has tiny `P(i)`, so `w_i` blows up; on the rare occasion I sample one, `w_i δ_i` could be huge and throw a wild gradient step. So before I even consider annealing, I should normalize the weights by their maximum over the replay memory, `w_i ← w_i / max_l w_l`, so the largest possible weight is 1 and every update is only ever scaled *downward*. That keeps the steps in a sane range and never amplifies. Because `max_l w_l` is a single scalar for the current replay memory, at `β = 1` it scales the whole expected update without moving its zero; it bounds the step size while preserving the corrected fixed point.

But do I want the *full* correction the whole time? Two reasons to soften it early. The unbiasedness argument I just made is about the converged fixed point — it matters at the *end* of training. Early on the whole process is wildly non-stationary anyway: the policy is changing, the induced state distribution is drifting, and the bootstrap targets `θ_target` keep shifting under me. Insisting on an exactly-unbiased gradient with respect to a fixed distribution, while that distribution is a moving target, is correcting for a precision I don't have. A small bias early is in the noise. And the full correction `w_i` actively *fights* the prioritization — it scales down precisely the high-priority transitions whose extra weight is what's buying me the early speedup. So I want little correction early (lean into the aggressive prioritization, learn fast) and full correction late (be unbiased where it counts, near convergence). Put a second exponent on the weight:

```
w_i = ( (1/N) · 1/P(i) )^β = ( N · P(i) )^{-β}.
```

At `β = 0`, `w_i = 1` — no correction, pure prioritized speed. At `β = 1` it's the full IS weight I derived above. So I anneal `β` from a starting `β_0 < 1` linearly up to `1`, reaching exactly 1 at the end of learning. Note `β` and `α` interact: cranking `α` prioritizes harder, cranking `β` corrects harder, and raising both together means "sample very aggressively but also undo it strongly" — they're a push-pull pair, and it's easy to retreat toward the baseline by lowering `α` or raising `β`.

And there's a bonus from the normalized weights I didn't expect. With non-linear function approximation the gradient is only a local, first-order picture of the loss surface; a big step walks off where that linearization is valid and can be destructive — the usual remedy is just a smaller global step-size. But here the structure does something nicer: prioritization guarantees a high-error transition is *seen many times*, while the IS weight (always ≤ 1 after normalization) *shrinks each individual step*. So instead of one big leap I get many small, re-linearized steps along the same direction — the Taylor expansion is re-approximated at each, so I can follow curvature instead of overshooting it. And as `β → 1` the normalizer `max_l w_l` grows, which steadily shrinks the average effective step — the annealing doubles as an implicit step-size decay. That's a free stabilizer.

Now the implementation has to be honest about cost, because at `N = 10^6` I cannot afford anything linear in `N` per sample. For proportional, I need to (a) sample an index with probability `p_i^α / Σ p_k^α` and (b) update a single `p_i^α` after each learning step, both fast. Store the `p_i^α` values at the leaves of a binary tree where every internal node holds the sum of its two children; the root holds `Σ_k p_k^α =: p_total`. Updating one leaf and walking up to fix the sums on the path to the root is `O(log N)`. To draw a sample: pick a target mass `u ∈ [0, p_total]`, then descend from the root — at each node, if the left child's sum is greater than `u` go left, else subtract the left sum from `u` and go right — landing at the leaf whose cumulative interval contains `u`. The equality case belongs on the right: if `u` is exactly the left cumulative boundary, the sampled mass has crossed that interval. That's a prefix-sum search, also `O(log N)`, and it lands on leaf `i` with probability exactly `p_i^α / p_total`, which is `P(i)`. For a minibatch of `k`, rather than `k` independent draws I split `[0, p_total]` into `k` equal mass segments and draw one uniform `u` from each. That's stratified sampling: it lowers the variance of the minibatch estimate and makes it less likely that a whole batch lands in one cumulative region, though if one transition owns enough mass it can still appear through multiple segments.

For the rank-based variant the implementation is different but the spirit is the same. I don't need the buffer perfectly sorted — `rank` is already only a coarse proxy — so I keep an array-based binary heap and treat the heap array as an approximately-sorted array, re-sorting it only rarely (once per `~10^6` steps) to stop it drifting too far out of order. Then the power-law CDF over ranks can be precomputed and chopped into `k` equal-probability segments (these only change when `N` or `α` changes); at sample time I pick one transition uniformly within each segment. Here the rank order makes the stratification closer to the "one high-rank, one medium-rank, one low-rank" intuition, and the sampling cost is essentially constant once the partitions are available.

One subtlety the normalization needs: to scale by `1/max_l w_l` I need the largest weight, and since `w` is decreasing in `P`, that's the transition with the *smallest* `P(i)`, i.e. the smallest `p_i^α`. So alongside the sum-tree I keep a parallel min-tree over the same `p_i^α` leaves, so `min_l p_l^α` (hence `P_min`, hence `max_l w_l = (N · P_min)^{-β}`) is also `O(log N)`. Clean.

Now assemble the whole agent. I'll build on Double DQN — the target `y = R + γ Q_target(S', argmax_a Q(S',a))` to keep the overestimation bias down — and change exactly the replay: sample from `P` instead of uniformly, fold `w_i` into the update as a per-transition multiplier, write the fresh `|δ_i| + ε` back as the new score, and insert every new transition at the current maximum score so it's guaranteed at least one replay. Per step I store the transition, and every `K` environment steps I do a minibatch: for each of the `k` samples, draw an index `i ∼ P`, compute its weight `w_i = (N·P(i))^{-β} / max_l w_l`, compute `δ_i = R_i + γ_i Q_target(S_i, argmax_a Q(S_i,a)) - Q(S_{i-1}, A_{i-1})`, set `p_i ← |δ_i| + ε`, and accumulate `Δ ← Δ + w_i · δ_i · ∇_θ Q(S_{i-1}, A_{i-1})`. Then `θ ← θ + η Δ`, reset `Δ`, and copy weights to the target network from time to time.

One hyperparameter has to move relative to the baseline. Prioritization deliberately picks high-`|δ|` transitions far more often than uniform would, so the typical gradient magnitude per minibatch goes *up*. If I keep the old step-size the effective learning rate is now too hot. So I cut `η` by a factor of 4. Beyond that, `α` and `β_0` are the only genuinely new knobs; a coarse search lands around `α = 0.7, β_0 = 0.5` for rank-based and `α = 0.6, β_0 = 0.4` for proportional — aggressive enough to get the speedup, corrected enough to stay stable.

A worry I should resolve about the two variants. I argued rank should be the more robust one — immune to outliers and error scale, heavy-tailed by construction. But the base agent already *clips* both rewards and TD errors to `[−1, 1]` for stability. Clipping is exactly an outlier-removal step: it caps the very magnitudes that would have made proportional misbehave. With the outliers already gone, proportional's main weakness is mostly neutralized, so I should not expect the two variants to separate only because of rare huge errors.

Let me put the core data structure and buffer down as real code. The sum-tree first:

```python
import operator
import random
import numpy as np

class SegmentTree:
    """Array-backed binary tree: internal node = `operation` of its children.
    O(log N) point-update and O(log N) range-reduce. Used to hold p_i**alpha."""
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0  # power of two
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        elif mid + 1 <= start:
            return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            return self._operation(
                self._reduce_helper(start, mid, 2 * node, node_start, mid),
                self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end))

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity            # to leaf
        self._value[idx] = val
        idx //= 2
        while idx >= 1:                  # repair sums/mins up to the root
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
        # descend: the prefix-sum search that turns a mass u into an index ~ P(i)
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:      # while not a leaf
            if self._value[2*idx] > prefixsum:
                idx = 2*idx              # target mass is in the left subtree
            else:
                prefixsum -= self._value[2*idx]   # skip the left mass, go right
                idx = 2*idx + 1
        return idx - self._capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, min, float('inf'))

    def min(self, start=0, end=None):
        return super().reduce(start, end)
```

And the replay buffer fills the sampling slot without changing the storage contract the uniform buffer already had:

```python
class ReplayBuffer:
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
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
        self._score_exponent = score_exponent     # alpha: uniform (0) <-> sharper scores
        cap = 1
        while cap < size:
            cap *= 2
        self._it_sum = SumSegmentTree(cap)        # holds score_i**alpha
        self._it_min = MinSegmentTree(cap)        # supplies the max correction weight
        self._max_score = 1.0                     # new transitions enter at the top

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        weighted_score = self._max_score ** self._score_exponent
        # unseen transition -> maximal score, so it is replayed at least once
        self._it_sum[idx] = weighted_score
        self._it_min[idx] = weighted_score

    def _sample_weighted(self, batch_size):
        res = []
        total = self._it_sum.sum(0, len(self._storage))
        seg = total / batch_size                # stratify: one draw per segment
        for i in range(batch_size):
            mass = random.random() * seg + i * seg
            res.append(self._it_sum.find_prefixsum_idx(mass))   # leaf ~ P(i)
        return res

    def sample(self, batch_size, correction_exponent):
        idxes = self._sample_weighted(batch_size)
        total = self._it_sum.sum(0, len(self._storage))
        # max weight comes from the least-probable transition (smallest P)
        p_min = self._it_min.min(0, len(self._storage)) / total
        max_weight = (p_min * len(self._storage)) ** (-correction_exponent)
        weights = []
        for idx in idxes:
            p_sample = self._it_sum[idx] / total               # P(i)
            w = (p_sample * len(self._storage)) ** (-correction_exponent)
            weights.append(w / max_weight)                      # normalize: w <= 1
        return self._encode_sample(idxes), np.array(weights, dtype=np.float32), idxes

    def update_scores(self, idxes, scores):
        # write back p_i = |delta_i| + epsilon, supplied by the learner
        assert len(idxes) == len(scores)
        for idx, score in zip(idxes, scores):
            assert score > 0
            assert 0 <= idx < len(self._storage)
            weighted_score = score ** self._score_exponent
            self._it_sum[idx] = weighted_score
            self._it_min[idx] = weighted_score
            self._max_score = max(self._max_score, score)
```

And the learning step that ties it together — sample from `P`, Double-DQN `δ`, reweight by `w`, write `|δ| + ε` back:

```python
def td_error(batch, online_net, target_net, gamma):
    s, a, r, s2, done = batch
    next_a = online_net(s2).argmax(axis=1)                       # Double DQN: select online
    y = r + (1.0 - done) * gamma * target_net(s2)[range(len(s2)), next_a]
    return y - online_net(s)[range(len(s)), a]                   # target minus current Q


def train_step(buffer, online_net, target_net, opt, batch_size,
               correction_exponent, gamma, score_floor=1e-6):
    batch, weights, idxes = buffer.sample(batch_size, correction_exponent)
    delta = td_error(batch, online_net, target_net, gamma)
    loss = (weights * huber(delta)).mean()                         # gradient gets the w_i factor
    opt.zero_grad(); loss.backward(); opt.step()                   # eta already cut by ~4x vs uniform
    buffer.update_scores(idxes, np.abs(delta) + score_floor)       # new score p_i = |delta_i| + eps
```

The chain, start to end: uniform replay spends updates on transitions the network already predicts, and in needle-in-a-haystack environments an oracle that replays in the right order is exponentially faster — so order is worth chasing. `|δ|` is a free, online proxy for how much a transition can still teach, but argmax-on-`|δ|` is brittle (stale unreplayed priorities, noise-spike chasing, diversity collapse). Softening it to `P(i) = p_i^α/Σ p_k^α` recovers uniform at `α=0` and greedy as `α→∞` while keeping full support, with `p_i=|δ_i|+ε` (proportional, sees magnitude) or `1/rank` (rank-based, robust, heavy-tailed). But sampling from `P` instead of `1/N` moves the fixed point SGD converges to, so I reweight each term by `w_i=((1/N)·1/P(i))^β` — which under `P` returns the expected update to the uniform one — normalized by `1/max_l w_l` for downward-only steps, and anneal `β:β_0→1` to leave the bias uncorrected early (where non-stationarity dominates and aggressive replay buys speed) and fully corrected near convergence. A sum-tree gives `O(log N)` sampling and update with stratified minibatches; a parallel min-tree supplies the normalizer. Drop this in for Double DQN's uniform sampler, cut `η` by 4 because the gradients are now larger, and that's the method.
