# Greedy Maximization of Submodular Functions

## Problem

Maximize a non-negative, monotone, submodular set function f: 2^V -> R>=0
subject to a cardinality constraint |S| <= k. This is NP-hard (it generalizes
maximum coverage), so we seek a polynomial-time algorithm with a provable
worst-case ratio against f(OPT).

- **Monotone:** A subset B => f(A) <= f(B).
- **Submodular (diminishing returns):** for A subset B, e not in B,
  f(A + e) - f(A) >= f(B + e) - f(B). The marginal gain f(e | S) = f(S+e) - f(S)
  is non-increasing as S grows.

## Key idea

Grow S one element at a time, each round adding the element of **maximum marginal
gain**:

    S <- empty
    repeat k times:  S <- S + argmax_{e not in S} [ f(S + e) - f(S) ]

That single rule -- argmax marginal gain -- is the whole method.

## Guarantee

Let S_i be the greedy set after i additions, O = OPT, |O| <= k. The greedy choice
beats adding any single o in O, so it beats the average:

    f(S_{i+1}) - f(S_i) >= (1/k) sum_{o in O} [ f(S_i + o) - f(S_i) ]
                         >= (1/k) [ f(S_i ∪ O) - f(S_i) ]    (submodularity)
                         >= (1/k) [ f(OPT) - f(S_i) ]        (monotonicity).

With gap a_i = f(OPT) - f(S_i) this is a_{i+1} <= (1 - 1/k) a_i, so after k rounds
a_k <= (1 - 1/k)^k a_0 <= (1/e) f(OPT), giving

    f(S_greedy) >= [ 1 - (1 - 1/k)^k ] f(OPT) >= (1 - 1/e) f(OPT) ≈ 0.632 f(OPT).

This constant is best possible: no polynomial algorithm beats (1 - 1/e) for
maximum coverage (P != NP; exponentially many queries in the value-oracle model).
The cardinality constraint is the uniform matroid -- the favorable special case;
under a general matroid the same greedy guarantees only 1/2.

## Acceleration (lazy greedy)

Because marginals are monotonically non-increasing, a previously computed gain is
an upper bound on the current gain. Keep gains in a max-heap; re-evaluate only the
top, and if its refreshed gain still tops the heap it is the true argmax this
round. Same output as naive greedy, far fewer oracle calls.

## Code

```python
import heapq


class SubmodularFunction:
    """Monotone submodular f over a ground set of indices, as a value oracle."""
    def __init__(self, ground_set):
        self.ground_set = list(ground_set)

    def value(self, S):
        raise NotImplementedError

    def marginal(self, e, S):
        return self.value(S | {e}) - self.value(S)


class MaxCoverage(SubmodularFunction):
    """f(S) = | union of the sets indexed by S |. Non-neg, monotone, submodular."""
    def __init__(self, sets):
        super().__init__(range(len(sets)))
        self.sets = [set(s) for s in sets]

    def value(self, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(covered)

    def marginal(self, e, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(self.sets[e] - covered)


def greedy(f, k):
    """Maximum marginal gain, k times.  f(S) >= (1 - 1/e) f(OPT)."""
    S = set()
    for _ in range(k):
        best_e, best_gain = None, -1.0
        for e in f.ground_set:
            if e in S:
                continue
            g = f.marginal(e, S)
            if g > best_gain:
                best_e, best_gain = e, g
        if best_e is None or best_gain <= 0:
            break
        S.add(best_e)
    return S


def lazy_greedy(f, k):
    """Minoux's accelerated greedy: same output, fewer oracle calls."""
    S = set()
    heap = []  # [-gain, element, size_when_evaluated]
    for e in f.ground_set:
        heapq.heappush(heap, [-f.marginal(e, S), e, 0])
    while len(S) < k and heap:
        neg_gain, e, evaluated_at = heap[0]
        if e in S:
            heapq.heappop(heap)
            continue
        if evaluated_at == len(S):
            heapq.heappop(heap)
            if -neg_gain <= 0:
                break
            S.add(e)
        else:
            heap[0][0] = -f.marginal(e, S)
            heap[0][2] = len(S)
            heapq.heapreplace(heap, heap[0])
    return S
```

On a small instance both selectors return the same set; `lazy_greedy` reaches it
with fewer marginal evaluations.
