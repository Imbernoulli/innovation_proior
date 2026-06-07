# Shannon information theory — the result

## Problem

Build a general theory of communication that (1) measures information independent of meaning, (2) gives the
exact limit to which a statistical source can be compressed, and (3) defines a noisy channel's capacity and
states exactly what rate and reliability are simultaneously achievable.

## Key idea

Treat a message as a *selection from a set of possible messages* — discard meaning. Model the source as an
ergodic stochastic (Markov) process. Then a single quantity, the **entropy** H = −Σ p log p, measures both
the information a source produces (its compression limit) and, through the **equivocation** (conditional
entropy of input given output), the toll noise charges. Both the source-coding limit and the noisy-channel
limit are governed by entropy.

## The entropy

For a set of probabilities p_1, …, p_n, require a measure H(p_1, …, p_n) that is (1) continuous in the p_i,
(2) for equal p_i = 1/n, a monotonically increasing function of n, and (3) invariant to staging a choice as
successive choices (grouping: H is the weighted sum of the staged H's). These three conditions force, uniquely,

    H = −K Σ_i p_i log p_i,    K > 0 a choice of unit.

With base-2 logs and K = 1, H is in **bits**. Properties: H = 0 iff one outcome is certain; H ≤ log n with
equality iff uniform; H(x, y) ≤ H(x) + H(y) with equality iff x, y are independent. This is the entropy of
Boltzmann/Gibbs statistical mechanics, now read as a measure of information.

**Uniqueness proof (sketch).** Let A(n) = H(1/n, …, 1/n). Grouping gives A(s^m) = m A(s); squeezing
s^m ≤ t^n < s^{m+1}, taking logs, and using monotonicity forces A(t) = K log t. For commensurable
p_i = n_i/Σn_j, decompose a uniform choice among Σn_j outcomes two ways:
A(Σn_j) = H(p) + Σ_i p_i A(n_i), i.e. K log(Σn_j) = H + K Σ p_i log n_i, so H = −K Σ p_i log p_i. Continuity
extends to incommensurable p_i. ∎

## Theorem 1 — source coding (compression limit)

For a source of entropy H bits/symbol: the length-N output sequences split, for large N, into a set of total
probability < ε and a **typical set** of ≈ 2^{HN} sequences each of probability ≈ 2^{−HN}. Equivalently
(1/N) log n(q) → H, where n(q) is the number of most-probable sequences accumulating probability q.
Consequently the source can be encoded with an average of H bits per symbol and no fewer; over a noiseless
channel of capacity C bits/second it can be transmitted at an average rate approaching C/H symbols/second,
and at no higher rate (since the channel-input entropy equals the source entropy and cannot exceed C).
**Entropy is the compression limit.** A prefix code assigning each message a codeword of length ≈ log(1/p)
achieves average length within one bit of H, → H per symbol on long blocks.

## Theorem 2 — noisy-channel coding (capacity)

Define the **equivocation** H_y(x) = conditional entropy of the input x given the output y (the uncertainty
about what was sent that remains after reception). The transmission rate is R = H(x) − H_y(x), and the
**channel capacity** is

    C = max over input sources of [ H(x) − H_y(x) ].

**Coding theorem.** If a source has entropy per second H ≤ C there exists a coding system transmitting it
over the channel with arbitrarily small error frequency. If H > C, the equivocation can be brought within ε of
H − C but no lower; no code gives equivocation below H − C.

**Proof of achievability — random coding (sketch).** Drive the channel with the maximizing source over blocks
of length T: about 2^{TH(x)} typical inputs, about 2^{TH(y)} typical outputs, and each output has a fan of
about 2^{TH_y(x)} reasonable input causes. To send 2^{TR} messages with R < C, assign them to typical inputs
*at random* and average the error over all such codebooks (existence, not construction). For a given output,
the probability a given input point is a message is 2^{T(R − H(x))}; the probability that no competing message
lands in the output's fan is

    P = (1 − 2^{T(R − H(x))})^{2^{TH_y(x)}}.

Since R < H(x) − H_y(x), write R − H(x) = −H_y(x) − η (η > 0); the expected number of competing messages
in the fan is 2^{TH_y(x)} · 2^{−TH_y(x) − Tη} = 2^{−Tη} → 0, so P → 1 and the error probability → 0 at the
fixed rate R. Below capacity the fans pack into the typical-input space without overlap (sphere-packing);
above it overlap is forced. Since the average error is small, some specific codebook achieves it. ∎

**Converse (sketch).** If a source of entropy C + a per unit time could be sent with equivocation a − ε,
then the delivered rate would be (C + a) − (a − ε) = C + ε > C, contradicting
C = max[H(x) − H_y(x)]. So rates above C cannot be made reliable. ∎

## Executable source-coding check

```python
import heapq
import math
from collections import Counter
from itertools import count

def empirical_distribution(symbols):
    n = len(symbols)
    if n == 0:
        raise ValueError("sample must be non-empty")
    return {s: c / n for s, c in Counter(symbols).items()}

def information_measure(distribution):
    """H = -sum p log2 p: bits/symbol, the compression limit."""
    if any(p < 0 for p in distribution.values()):
        raise ValueError("probabilities must be non-negative")
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)

def prefix_code(distribution):
    """Binary prefix code made by repeatedly merging the two least likely subtrees."""
    serial = count()
    heap = []
    for sym, p in distribution.items():
        if p < 0:
            raise ValueError("probabilities must be non-negative")
        if p > 0:
            heap.append([p, next(serial), {sym: ""}])
    if not heap:
        raise ValueError("distribution must have positive mass")
    heapq.heapify(heap)
    if len(heap) == 1:
        return heap[0][2]
    while len(heap) > 1:
        p_lo, _, lo = heapq.heappop(heap)
        p_hi, _, hi = heapq.heappop(heap)
        merged = {sym: "0" + bits for sym, bits in lo.items()}
        merged.update({sym: "1" + bits for sym, bits in hi.items()})
        heapq.heappush(heap, [p_lo + p_hi, next(serial), merged])
    return heap[0][2]

def average_length(code, distribution):
    return sum(p * len(code[s]) for s, p in distribution.items() if p > 0)

def source_coding_check(symbols):
    dist = empirical_distribution(symbols)
    H = information_measure(dist)
    code = prefix_code(dist)
    L = average_length(code, dist)
    return H, L, code

if __name__ == "__main__":
    text = "the source coding limit is the entropy of the source"
    H, L, code = source_coding_check(text)
    print(f"H = {H:.3f} bits/symbol; avg length = {L:.3f} bits/symbol")
    assert L + 1e-12 >= H
    assert L < H + 1 + 1e-12
```

## Why it matters

Entropy unifies the two limits of communication: it is the floor on compression (source coding) and, through
capacity C = max[H(x) − H_y(x)], the ceiling on reliable transmission (channel coding). The noisy-channel
theorem's surprise is that error can be driven to zero at a *fixed* positive rate up to C — redundancy need
not grow without bound — proved not by building a code but by averaging over random codes.
