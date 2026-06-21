We are handed a stream of records that arrives strictly in order and can be read only once — rows off a sequential scan, events on a wire, records on a tape of indeterminate length — and we want a simple random sample of exactly $k$ of them, drawn without replacement so that every size-$k$ subset is equally likely. Two facts make this awkward. The total length $N$ is not known in advance, and learning it would cost an extra pass we cannot afford. And $N$ may dwarf $k$, so we cannot buffer everything and sample at the end: the working memory must stay $O(k)$, independent of $N$. The classical sampling toolkit is no help here, because every piece of it feeds on $N$. Picking $k$ distinct indices from $\{1,\dots,N\}$ needs $N$ as the modulus; the sequential rule that takes the $i$-th record with probability $(k-\text{taken})/(N-i+1)$ needs $N$ in the denominator; the skip-gap generators need $N$ as a distribution parameter. Knock $N$ out and the whole construction collapses. The obvious one-pass dodges fail too: keeping each record independently with a fixed probability $p$ is one pass and $O(1)$ per record, but it yields a $\mathrm{Binomial}(N,p)$ number of records — random rather than exactly $k$, with a mean $Np$ we cannot even aim at $k$ without knowing $N$. And buffering the whole stream to sample at the end is trivially correct but $O(N)$ memory, the disqualifying flaw when $N \gg k$. So the hardness is not the sampling, it is that we are blind to the size.

What resolves it is to stop trying to reference $N$ at all and instead demand a single local invariant: after $t$ items have passed, the $k$ items held are a uniform sample of those $t$. If that holds at every step, then when the stream ends at $t=N$ we automatically have a uniform sample of $N$, and we never needed to know $N$ — we just kept patching the invariant forward. I propose reservoir sampling. Seed the reservoir with the first $k$ records, which are trivially a uniform sample of themselves. When the $(t+1)$-st record arrives, the invariant pins down the rule with no freedom left: in a true uniform size-$k$ sample of $t+1$ items, every item is equally likely to be among the chosen $k$, so each — including the newest — is present with probability $k/(t+1)$. Therefore the new record must be admitted with probability exactly $k/(t+1)$; anything else gives it the wrong inclusion probability and breaks the invariant. If admitted, it must displace one of the $k$ current holders, and to keep the survivors exchangeable we evict a uniformly random slot. The cleanest realization of this is Algorithm R (Waterman): when the $i$-th record arrives, draw an integer $j$ uniform in $\{1,\dots,i\}$, and if $j \le k$ overwrite slot $j$. The single draw does both jobs at once — it admits with probability $k/i$, and conditioned on admission $j$ is uniform over the $k$ slots, which is exactly the uniform eviction.

The invariant propagates by induction, and it is worth checking that it does so not only on marginals but on the full subset law. For the marginal, suppose after $t$ records each seen item sits in the buffer with probability $k/t$. An old item $x$ survives the $(t+1)$-st step in two disjoint ways: the new record is rejected, probability $1-k/(t+1)$, and nothing moves; or it is admitted, probability $k/(t+1)$, but the evicted slot is not $x$'s, probability $1-1/k$. Conditioning on $x$ having been in,
$$\Pr[x \text{ in}] = \frac{k}{t}\left[\left(1-\frac{k}{t+1}\right) + \frac{k}{t+1}\left(1-\frac1k\right)\right] = \frac{k}{t}\left[1-\frac{1}{t+1}\right] = \frac{k}{t}\cdot\frac{t}{t+1} = \frac{k}{t+1},$$
since the $\pm k/(t+1)$ cancel and $(k/(t+1))(1/k)=1/(t+1)$. The new record is in with probability $k/(t+1)$ by construction, so all $t+1$ items land at the right marginal. The stronger claim is that every fixed size-$k$ subset $A$ of the first $t+1$ items has probability $1/\binom{t+1}{k}$. If $A$ excludes the new record, the only path to $A$ is that the old buffer was exactly $A$ and the new record was rejected, with probability $\frac{1}{\binom{t}{k}}(1-\frac{k}{t+1}) = \frac{1}{\binom{t}{k}}\cdot\frac{t+1-k}{t+1} = \frac{1}{\binom{t+1}{k}}$. If $A$ includes the new record, its $k-1$ old members can be paired with any of $t-k+1$ old extras before the step; each such buffer has probability $1/\binom{t}{k}$, and then the new record must be accepted and that extra evicted, probability $(k/(t+1))(1/k)=1/(t+1)$, giving $\frac{t-k+1}{\binom{t}{k}(t+1)} = \frac{1}{\binom{t+1}{k}}$. Base case $t=k$ holds, and the induction carries the subset law all the way to $t=N$ without ever touching $N$.

Algorithm R is correct but wasteful in randomness. For every one of the $N$ records it draws an integer and compares, $\Theta(N)$ draws, yet almost none of those records ever enter the reservoir. The $i$-th enters with probability $k/i$, so the expected number of admissions over $i=k+1,\dots,N$ is $\sum_{i=k+1}^{N} k/i = k(H_N - H_k) \approx k\ln(N/k)$, where $H_m$ is the $m$-th harmonic number; including the initial fill the expected count of records that ever enter is $k(1+H_N-H_k)$. For $N$ a billion and $k$ a thousand that is a few thousand admissions against a billion records, and on real hardware a good uniform draw is expensive enough to dominate. The right move is to spend randomness only at the admissions: decide in one shot how far ahead the next admission lands, leap over the rejected run without drawing anything for it. The freedom to do this is exactly what is left over — uniformity forces the per-step acceptance probability $k/(t+1)$, but not how we realize the sequence of accept/reject decisions.

To get a closed-form skip I switch to the equivalent random-key view, which is Algorithm L. Give each record an iid tag $u_i \sim U(0,1)$ and keep the $k$ records with the smallest tags; since the tags are iid with no ties, every $k$-subset is equally likely to be the smallest-$k$, and nothing in that statement mentions $N$. Online I need only the $k$ smallest tags, and the only number governing the next admission is the largest of those, the threshold $w$ — a new record is admitted exactly when its tag falls below $w$. So I track just the scalar $w$. After the reservoir fills, $w$ is the maximum of $k$ iid $U(0,1)$ tags, whose CDF is $\Pr[\max \le x]=x^k$; by inverse transform a single draw gives $w = U^{1/k} = \exp(\log U / k)$. With $w$ held fixed across a run, each passing record is admitted independently with probability $w$, so the number skipped before the next admission is geometric, $\Pr[\text{skip}=g]=(1-w)^g w$, and inverse transform gives it in one shot as
$$\text{gap} = \left\lfloor \frac{\log U}{\log(1-w)} \right\rfloor,$$
one $\log$, one division, one floor — constant work, one draw — to leap an entire run of rejects (the admitted position sits one past the skipped records). At each admission the tag of the new record lies below $w$, so it joins the smallest $k$ and the old threshold-holder is evicted; the retained tags are each then uniform on $(0, w_{\text{old}})$, and their new maximum is $w_{\text{old}}$ times the maximum of $k$ uniforms, so the threshold tightens multiplicatively as $w_{\text{new}} = w_{\text{old}} \cdot U^{1/k}$. The threshold marches monotonically downward and admissions thin out, mirroring $k/i$ shrinking as $i$ grows, but now without ever computing $k/i$ or touching a rejected record. The tags were only a device to generate the right admission times; the subset proof already fixed the replacement, so on admission we still drop the new record into a uniformly random slot. The result has the optimal-order $O(k(1+\log(N/k)))$ random draws.

The same key construction generalizes to weighted sampling without replacement. If records carry weights $w_i$ and we want an item's chance of selection to scale with its weight, bend the tag to the key $\text{key}_i = u_i^{1/w_i}$ with $u_i \sim U(0,1)$, and keep the $k$ largest keys. Its law is $\Pr[\text{key}_i \le x] = \Pr[u_i \le x^{w_i}] = x^{w_i}$ for $x \in (0,1)$, so a larger $w_i$ pushes the key toward $1$ — heavy items get systematically larger keys. For a single draw the probability that item $i$ holds the maximum key is
$$\int_0^1 \frac{d}{dx}\,x^{w_i} \prod_{j\ne i} x^{w_j}\,dx = \int_0^1 w_i\,x^{w_i-1}\,x^{W-w_i}\,dx = w_i\int_0^1 x^{W-1}\,dx = \frac{w_i}{W},\qquad W=\sum_j w_j,$$
exactly proportional to weight, and keeping the top $k$ gives the without-replacement version since each subsequent round conditions on the remaining weights. Online this is A-Res: a size-$k$ min-heap whose minimum is the threshold $T$, admitting an arriving $(item, w)$ with $\text{key}=u^{1/w}$ whenever $\text{key} > T$. The heap is the weighted analogue of the single scalar $w$ from the uniform case — it must track $k$ separate thresholds because the keys are no longer order statistics of a common uniform. The skip-the-rejects speedup carries over as A-ExpJ, except the budget until the next admission accumulates in weight, not count: with threshold $T_w$, draw $r \sim U(0,1)$ and set an exponential weight budget $X_w = \log(r)/\log(T_w)$ (both logs negative, so $X_w > 0$), walk the stream subtracting weights until the budget is exhausted, admit the crossing item with a key drawn conditioned on clearing $T_w$ (cutoff $T_w^{w_i}$, then $r_2 \sim U(\text{cutoff},1)$, key $= r_2^{1/w_i}$), refresh $T_w$ to the new heap min, and redraw the budget — dropping the draw count from $O(n)$ to $O(k\log(n/k))$. The basic per-item rule remains the one to use when $N$ is not huge and the draw count does not dominate; the gap-driven and weighted versions fill the same bounded-buffer interface.

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
