# Reservoir Sampling

## Problem

Draw a simple random sample of exactly `k` items, without replacement and with every
size-`k` subset equally likely, from a stream that is read **once, front to back**, whose
length `N` is **unknown in advance** and may be far larger than `k`. The algorithm must use
`O(k)` working memory (independent of `N`) and never needs to know or count `N`.

## Key idea

Maintain the invariant *after `t` items have passed, the `k` items held are a uniform sample
of those `t`*. The invariant alone forces the rule: in a uniform size-`k` sample of `t+1`
items, each item — including the newest — is present with probability `k/(t+1)`, so the
`(t+1)`-st item must be admitted with probability exactly `k/(t+1)`, evicting a uniformly
random one of the `k` held items. Seed with the first `k` (a uniform sample of themselves);
induction carries the invariant to `t = N`, where each item is in the final sample with
probability `k/N` — all without referencing `N`.

A speedup follows from noticing later admissions are *rare*: after the first `k` items, their
expected count is `k(H_N - H_k) = k ln(N/k) + O(k)`. Including the initial fill, the expected
number of records that ever enter the buffer is `k(1 + H_N - H_k)`. So instead of spending a
random draw per item, generate the **number of items to skip** before the next admission
directly, and leap over the rejected run. With the equivalent "random key" view (give item
`i` a key `u_i ~ U(0,1)`, keep the `k` smallest keys), the admission threshold is a single
scalar `w` distributed as `U^{1/k}`; with `w` fixed across a run, admissions are independent
Bernoulli(`w`), so the gap is geometric and is drawn in `O(1)` as
`floor(log(U)/log(1-w))`. This is **Algorithm L**, with optimal-order
`O(k(1 + log(N/k)))` random draws.

## Algorithms

**Algorithm R (Waterman; per-item, `O(N)` draws).** Keep the first `k`; for the `i`-th item
(`i > k`) draw `j` uniform in `{1,...,i}` and if `j <= k` overwrite slot `j`. The single draw
both admits with probability `k/i` and picks the uniform eviction slot.

```
ReservoirSample(S[1..n], R[1..k])
  for i := 1 to k:        R[i] := S[i]
  for i := k+1 to n:
      j := randomInteger(1, i)
      if j <= k:          R[j] := S[i]
```

**Algorithm L (skip-accelerated; `O(k(1+log(N/k)))` draws).** `W = U^{1/k}` is the current
threshold; the next admission is `floor(log(U)/log(1-W)) + 1` positions ahead; on admission,
overwrite a random slot and tighten `W := W · U^{1/k}`.

```
ReservoirSample(S[1..n], R[1..k])
  for i := 1 to k:        R[i] := S[i]
  W := exp(log(random())/k)
  while i <= n:
      i := i + floor(log(random())/log(1-W)) + 1
      if i <= n:
          R[randomInteger(1,k)] := S[i]
          W := W * exp(log(random())/k)
```

**Why each design choice.** Seed with the first `k`: they are trivially a uniform sample of
themselves. Admit with `k/(t+1)`: the exact probability the new item belongs in a size-`k`
sample of `t+1` items — anything else breaks the invariant. Evict uniformly: keeps the held
items exchangeable so the survivor probabilities stay `k/(t+1)` (verified by the induction).
Skip-variate: per-item coin flips are pure waste once admissions are seen to be only
`~k ln(N/k)`; generate the gap instead. Geometric gap with `w = U^{1/k}`: tracking the single
threshold scalar (not the whole key array) is what gives `O(k)` memory and `O(1)` work per
admission.

## Correctness (Algorithm R, induction)

Claim: after `t` items each seen item is in the reservoir with probability `k/t`. Base
`t = k`: all `k` present, `k/k = 1`. Step: an old item `x` (in with prob `k/t`) survives the
`(t+1)`-st step iff the new item is rejected (prob `1 - k/(t+1)`) or admitted but does not
evict `x` (prob `(k/(t+1))(1 - 1/k)`):

```
Pr[x in] = (k/t) · [ (1 - k/(t+1)) + (k/(t+1))(1 - 1/k) ]
         = (k/t) · [ 1 - 1/(t+1) ]
         = (k/t) · t/(t+1) = k/(t+1).
```

The new item is in with prob `k/(t+1)` by construction, so all `t+1` items are at `k/(t+1)`.
The survivor bracket is exactly
`(1-k/(t+1)) + (k/(t+1))(1-1/k) = 1 - 1/(t+1)`.

The stronger subset induction also holds. If a fixed size-`k` subset `A` of the first `t+1`
items excludes the new item, then
`Pr[A] = (1/C(t,k)) * (1-k/(t+1)) = 1/C(t+1,k)`. If `A` includes the new item, its `k-1`
old items can be paired with any of `t-k+1` old extras before the step; accepting the new
item and evicting that extra has probability `1/(t+1)`, so
`Pr[A] = (t-k+1)/(C(t,k)(t+1)) = 1/C(t+1,k)`. Thus every size-`k` subset is equally likely.

For the exact skip distribution, if `S` is the number skipped after `t` processed records,
then `S > s` means the next `s+1` records are rejected:

```
F(s) = Pr[S <= s]
     = 1 - prod_{m=1}^{s+1} (1 - k/(t+m))
     = 1 - prod_{m=1}^{s+1} (t+m-k)/(t+m).
```

Thus
`F(s) = 1 - [((t+s+1-k)!/(t-k)!) * (t!/(t+s+1)!)]`, which is the distribution inverted by
the skip-based algorithms.

## Weighted, without replacement (A-Res / A-ExpJ)

Generalize the key to `key_i = u_i^{1/w_i}`, `u_i ~ U(0,1)`; keep the `k` **largest** keys.
Then `Pr[key_i <= x] = x^{w_i}`, so for a single draw item `i` has the largest key with
probability `∫_0^1 w_i x^{w_i-1} x^{W-w_i} dx = w_i / W` where `W = Σ_j w_j` — proportional to
weight. **A-Res:** size-`k` min-heap, threshold `T` = min key, admit when `key > T`. **A-ExpJ:**
replace the per-item draw with an exponential **weight** budget `X_w = log(r)/log(T_w)`; walk
the stream subtracting weights until `X_w` is exhausted, admit the crossing item with a key
drawn conditioned on `> T_w`, refresh `T_w`, redraw the budget — `O(k log(n/k))` draws.

## Working code

```python
import math, random, heapq

# Algorithm R: per-item decision, O(n) random draws.
class StreamSampler:
    def __init__(self, k):
        self.k = k
        self.buffer = []
        self.i = 0

    def add(self, item):
        self.i += 1
        if len(self.buffer) < self.k:
            self.buffer.append(item)
        else:
            j = random.randrange(self.i)          # uniform in {0,...,i-1}
            if j < self.k:                        # admit with probability k/i
                self.buffer[j] = item

    def sample(self):
        return self.buffer

def sample_stream(stream, k):
    sampler = StreamSampler(k)
    for item in stream:
        sampler.add(item)
    return sampler.sample()

# Algorithm L: geometric gaps; O(k(1+log(n/k))) random draws.
class FastStreamSampler:
    def __init__(self, k):
        self.k = k
        self.buffer = []
        self.i = 0
        self.schedule_state = 1.0
        self.next_i = None

    def _schedule_next(self):
        gap = math.floor(math.log(random.random()) / math.log(1.0 - self.schedule_state))
        self.next_i += gap + 1

    def add(self, item):
        self.i += 1
        if len(self.buffer) < self.k:
            self.buffer.append(item)
            if len(self.buffer) == self.k:
                self.schedule_state = math.exp(math.log(random.random()) / self.k)
                self.next_i = self.k
                self._schedule_next()
        elif self.i == self.next_i:
            self.buffer[random.randrange(self.k)] = item
            self.schedule_state *= math.exp(math.log(random.random()) / self.k)
            self._schedule_next()

    def sample(self):
        return self.buffer

def sample_stream_fast(stream, k):
    sampler = FastStreamSampler(k)
    for item in stream:
        sampler.add(item)
    return sampler.sample()

# A-Res: key = u^(1/w), keep the k largest keys in a min-heap.
def sample_stream_weighted(stream, k):
    heap = []
    for item, weight in stream:
        key = random.random() ** (1.0 / weight)
        if len(heap) < k:
            heapq.heappush(heap, (key, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))
    return [item for _key, item in heap]

# A-ExpJ: weighted exponential jumps; O(k log(n/k)) random draws.
def sample_stream_weighted_fast(stream, k):
    it = iter(stream)
    heap = []
    for _ in range(k):
        item, weight = next(it)
        heapq.heappush(heap, (random.random() ** (1.0 / weight), item))
    threshold = heap[0][0]
    budget = math.log(random.random()) / math.log(threshold)
    for item, weight in it:
        budget -= weight
        if budget <= 0:
            cutoff = threshold ** weight
            key = random.uniform(cutoff, 1.0) ** (1.0 / weight)
            heapq.heapreplace(heap, (key, item))
            threshold = heap[0][0]
            budget = math.log(random.random()) / math.log(threshold)
    return [item for _key, item in heap]
```
