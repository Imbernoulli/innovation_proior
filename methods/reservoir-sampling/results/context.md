# Context: one-pass random sampling from a stream of unknown length

## Research question

We are handed a sequence of records that arrives strictly in order and can be read only
once — records on a magnetic tape of indeterminate length, rows streaming off a sequential
scan, events on a wire. We want a *simple random sample* of exactly `k` of them, without
replacement, with every size-`k` subset equally likely. Two facts make this hard. First, the
total length `N` is **not known in advance**, and determining it would cost an extra pass we
cannot afford (rewinding the tape, re-scanning the file). Second, `N` may be astronomically
larger than `k`, so we cannot buffer the records and sample at the end: the working memory
must stay `O(k)`, independent of `N`. So the precise goal is an algorithm that, reading the
stream once and front to back, holds at most `O(k)` records in memory at any instant, never
needs to know or count `N`, and at the moment the stream ends emits `k` records that form a
uniform sample of *whatever* the stream turned out to be. A secondary goal, once a correct
method exists, is to make it *fast* — in particular to minimize the number of expensive
random-number draws, which on real machines can dominate the cost.

## Background

The classical sampling problem assumes `N` is known. With `N` in hand the problem is easy and
well studied (Knuth, *The Art of Computer Programming*, vol. 2, 1981; Vitter, "Faster methods
for random sampling", *CACM* 27, 1984): one can draw `k` distinct indices from `{1,...,N}` —
by a partial Fisher–Yates shuffle, by sequential selection where the `i`-th record is taken
with probability `(k - chosen)/(N - i + 1)`, or by generating skip gaps from the known
hypergeometric/beta structure. All of these consume `N` somewhere: either as the modulus for
index generation or as the parameter of the skip distribution.

The load-bearing facts we get to use:

- **Sampling without replacement as order statistics.** If we attach to each item an
  independent uniform key `u_i ~ U(0,1)`, then the items holding the `k` smallest keys are a
  uniform random `k`-subset — every subset is equally likely because the keys are iid and
  ties have probability zero. This converts "choose a random subset" into "keep the
  extreme-`k` of a stream of iid reals", which is something a one-pass scan *can* maintain.
- **Order statistics of uniforms are closed-form.** The maximum of `k` iid `U(0,1)` variables
  has CDF `Pr[max <= x] = x^k`. By inverse transform, a draw of that maximum is `U^{1/k}` for
  a single `U ~ U(0,1)`. More generally the `k`-th smallest of many uniforms has a Beta
  distribution. These let us reason about a "threshold" key in closed form.
- **Geometric gaps from a constant acceptance probability.** If, with a *fixed* per-item
  acceptance probability `p`, we scan items independently, the number skipped before the next
  acceptance is geometric: `Pr[gap = g] = (1-p)^g p`. Its inverse CDF gives a gap directly as
  `floor( log(U)/log(1-p) )` — so a whole run of rejections can be jumped over with a single
  uniform draw instead of one draw per item.
- **The harmonic-sum count of changes.** If the `i`-th item is accepted with probability
  `k/i`, the expected number of acceptances across `i = k+1,...,N` is
  `k(H_N - H_k) = k * ln(N/k) + O(k)`, where `H_m = sum_{i<=m} 1/i`. Acceptances are
  therefore *rare* relative to `N` — only logarithmically many — which is the structural hint
  that per-item work is wasteful.

The pain point that frames everything: any method that does a constant amount of work *per
record* must do `Theta(N)` work and draw `Theta(N)` random numbers, even though only
`~k ln(N/k)` records ever actually enter the sample. On the hardware of the time, generating
a high-quality uniform variate is expensive enough that this `Theta(N)` draw count is the
real bottleneck, not the I/O.

## Baselines

- **Two-pass / known-`N` sampling (Knuth 1981; Vitter 1984).** First pass counts `N`; second
  pass runs a standard known-`N` selection (sequential or skip-based). Correct and even fast
  per pass, but it needs *two* passes and a stored or recomputable `N`. The gap it leaves: on
  a one-pass medium of unknown length the first pass is exactly the thing we are forbidden to
  do.

- **Fixed-probability Bernoulli sampling.** Keep each record independently with some
  probability `p`. One pass, `O(1)` decision per record. But the sample size is
  `Binomial(N, p)` — random, not exactly `k`, and its mean `Np` cannot even be set without
  knowing `N`. The gap: it cannot deliver *exactly* `k`, and it cannot be tuned without `N`.

- **Buffer-everything-then-sample.** Read the whole stream into memory, then draw `k`. Trivially
  one pass and exactly `k`, but `O(N)` memory — the disqualifying flaw when `N >> k`.

- **Weighted sampling with known weights (Olken 1993; the Alias method, partial-sum trees,
  acceptance/rejection — see Efraimidis & Spirakis's survey).** When items carry weights `w_i`
  and we want a draw with probability proportional to weight, these classical methods build a
  global data structure (an alias table, a Fenwick tree of cumulative weights) over the *whole*
  population first. The gap: they need the full weight set up front and `O(N)` structure, so
  they are not one-pass and not streaming.

## Evaluation settings

The natural yardstick is correctness plus cost. *Correctness*: over many independent runs on a
fixed stream of length `N`, each item's empirical inclusion frequency should approach `k/N`,
and more strongly every `k`-subset should be equally likely (checkable on small `N` by
comparing subset frequencies to `1 / C(N, k)`). For the weighted variant, an item's
single-draw inclusion frequency should approach `w_i / sum_j w_j`. *Cost*: peak memory
(target `O(k)`), number of passes (target one), wall-clock CPU time, and — the metric the
speed work turns on — the **count of uniform random variates drawn** as a function of `N` and
`k`. Test streams are simple integer ranges or tagged records; weighted tests attach a known
weight to each item. The CPU-time measurements of the era were taken on sequential files large
enough that `N >> k`.

## Code framework

A streaming harness already exists: items arrive one at a time, we hold a small buffer, and we
emit the buffer at end-of-stream. The uniform RNG, the array, and (for the weighted case) a
binary heap are all standard primitives. What is missing is the *selection rule* — which
arriving items enter the buffer and which buffered item they displace — and the *acceptance
schedule* that decides this without knowing `N`. The stubs below mark exactly those holes.

```python
import math, random, heapq

class StreamSampler:
    """One pass, O(k) memory. Feed items with add(); read sample() at the end."""
    def __init__(self, k):
        self.k = k
        self.reservoir = []   # the O(k) buffer
        self.i = 0            # items seen so far (we never assume we know the final N)

    def add(self, item):
        self.i += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)        # seed phase: the first k are a valid size-k sample of themselves
        else:
            # TODO: decide whether this item enters, and which slot it displaces,
            #       so that the buffer stays a uniform size-k sample of all i items.
            pass

    def sample(self):
        return self.reservoir

def sample_stream(stream, k):
    s = StreamSampler(k)
    for x in stream:
        s.add(x)
    return s.sample()

def sample_stream_fast(stream, k):
    # TODO: same output as sample_stream, but skip whole runs of rejected items
    #       with a single random draw instead of one draw per item.
    pass

def sample_stream_weighted(stream, k):
    # stream yields (item, weight). Want each item kept with probability tied to its weight,
    # without replacement, in one pass and O(k) memory.
    # TODO: a per-item key built from weight + a min-heap threshold rule.
    pass
```
