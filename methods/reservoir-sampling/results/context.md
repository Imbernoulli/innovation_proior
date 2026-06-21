# Context: one-pass random sampling from a stream of unknown length

## Research question

We are handed a sequence of records that arrives strictly in order and can be read only
once — records on a magnetic tape of indeterminate length, rows streaming off a sequential
scan, events on a wire. We want a *simple random sample* of exactly `k` of them, without
replacement, with every size-`k` subset equally likely. The total length `N` is not known
in advance, and determining it would cost an extra pass. `N` may also be astronomically
larger than `k`, so working memory must stay `O(k)`, independent of `N`. The question is
how to select such a sample reading the stream once, front to back, while holding at most
`O(k)` records in memory at any instant.

## Background

The classical sampling problem assumes `N` is known. With `N` in hand the problem is well
studied (Knuth, *The Art of Computer Programming*, vol. 2, 1981; Vitter, "Faster methods
for random sampling", *CACM* 27, 1984): one can draw `k` distinct indices from `{1,...,N}` —
by a partial Fisher–Yates shuffle, by sequential selection where the `i`-th record is taken
with probability `(k - chosen)/(N - i + 1)`, or by generating skip gaps from the known
hypergeometric/beta structure. All of these consume `N` somewhere: either as the modulus for
index generation or as the parameter of the skip distribution.

Some standard facts about uniform variates that are available off the shelf:

- **Order statistics of uniforms are closed-form.** The maximum of `k` iid `U(0,1)` variables
  has CDF `Pr[max <= x] = x^k`. By inverse transform, a draw of that maximum is `U^{1/k}` for
  a single `U ~ U(0,1)`. More generally the `k`-th smallest of many uniforms has a Beta
  distribution.
- **The geometric distribution by inverse transform.** If a fair test with a *fixed* success
  probability `p` is repeated independently, the number of failures before the first success
  is geometric: `Pr[gap = g] = (1-p)^g p`. Its inverse CDF gives a gap directly as
  `floor( log(U)/log(1-p) )` from a single uniform draw `U`.
- **The harmonic numbers.** `H_m = sum_{i<=m} 1/i`, and `H_N - H_k = ln(N/k) + O(1/k)`. Sums
  of terms of the form `1/i` over a range therefore grow only logarithmically in the range.

## Baselines

- **Two-pass / known-`N` sampling (Knuth 1981; Vitter 1984).** First pass counts `N`; second
  pass runs a standard known-`N` selection (sequential or skip-based). Correct and even fast
  per pass, requiring `N` as a stored or recomputable value.

- **Fixed-probability Bernoulli sampling.** Keep each record independently with some
  probability `p`. One pass, `O(1)` decision per record. The sample size is
  `Binomial(N, p)` — random, not exactly `k`, and its mean `Np` depends on `N`.

- **Buffer-everything-then-sample.** Read the whole stream into memory, then draw `k`. One
  pass and exactly `k`, using `O(N)` memory.

- **Weighted sampling with known weights (Olken 1993; the Alias method, partial-sum trees,
  acceptance/rejection — see Efraimidis & Spirakis's survey).** When items carry weights `w_i`
  and we want a draw with probability proportional to weight, these classical methods build a
  global data structure (an alias table, a Fenwick tree of cumulative weights) over the *whole*
  population first, requiring the full weight set and `O(N)` structure up front.

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
emit the buffer at end-of-stream. The uniform RNG, arrays, iterators, and binary heaps are all
standard primitives.

```python
import math, random, heapq

class StreamSampler:
    """One pass, O(k) memory. Feed items with add(); read sample() at the end."""
    def __init__(self, k):
        self.k = k
        self.buffer = []
        self.i = 0

    def add(self, item):
        self.i += 1
        if len(self.buffer) < self.k:
            self.buffer.append(item)
        else:
            # TODO: decide whether this item enters and which buffered slot it displaces.
            pass

    def sample(self):
        return self.buffer

def sample_stream(stream, k):
    s = StreamSampler(k)
    for x in stream:
        s.add(x)
    return s.sample()

class FastStreamSampler:
    """Same input/output contract as StreamSampler; a faster internal strategy."""
    def __init__(self, k):
        self.k = k
        self.buffer = []
        self.i = 0

    def add(self, item):
        self.i += 1
        if len(self.buffer) < self.k:
            self.buffer.append(item)
        else:
            # TODO
            pass

    def sample(self):
        return self.buffer

def sample_stream_fast(stream, k):
    s = FastStreamSampler(k)
    for x in stream:
        s.add(x)
    return s.sample()

def sample_stream_weighted(stream, k):
    heap = []
    for item, weight in stream:
        # TODO
        pass
    return [item for _score, item in heap]

def sample_stream_weighted_fast(stream, k):
    # TODO
    pass
```
