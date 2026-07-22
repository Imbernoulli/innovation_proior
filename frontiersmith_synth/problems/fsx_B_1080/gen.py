#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of Fixed-ROM Codeword Binding to stdout.
Deterministic: all randomness seeded from testId only.
"""
import sys
import random
import heapq


def build_tree_shape(n, rng):
    """Build a genuinely SKEWED full binary tree with exactly n leaves: run a
    standard bottom-up Huffman merge over an independently-drawn power-law
    'legacy' weight vector (unrelated to the instance's real f[]). This gives
    a wide spread of leaf depths (a few short codewords, many long ones) --
    the ROM layout looks like it was designed for some OTHER distribution,
    not the one actually in play. Kraft equality (sum 2**-d == 1) holds
    exactly because it is a genuine full binary tree."""
    legacy_alpha = rng.uniform(6.0, 14.0)
    weights = []
    for rank in range(1, n + 1):
        raw = 30 * ((n - rank + 1) ** legacy_alpha)
        jitter = rng.uniform(0.8, 1.2)
        weights.append(max(1, raw * jitter))
    rng.shuffle(weights)

    heap = []
    for i, w in enumerate(weights):
        heapq.heappush(heap, (w, i, ("leaf", i)))
    counter = n
    while len(heap) > 1:
        w1, _, node1 = heapq.heappop(heap)
        w2, _, node2 = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, counter, ("node", node1, node2)))
        counter += 1
    _, _, root = heap[0]

    depths = [0] * n
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        if node[0] == "leaf":
            depths[node[1]] = d
        else:
            stack.append((node[1], d + 1))
            stack.append((node[2], d + 1))
    return depths


def gen_frequencies(n, rng, alpha):
    """Power-law-ish 'true rank' frequencies, rank 1 = largest. Returns list
    indexed by rank (0 = largest). Normalized to the instance's own size n (as
    a *fraction* of n, not a raw rank power) so the exponent cannot saturate
    the cap for large alpha/n and collapse many distinct ranks onto the same
    clipped value -- every rank gets its own, strictly-decreasing magnitude."""
    max_f = 2_000_000
    freqs = []
    for rank in range(1, n + 1):
        frac = (n - rank + 1) / n           # in (0, 1], 1.0 only at rank 1
        raw = max_f * (frac ** alpha)
        jitter = rng.uniform(0.9, 1.1)
        v = max(1, int(round(raw * jitter)))
        v = min(v, max_f)                    # safety net only (jitter <=1.1x)
        freqs.append(v)
    # guarantee strict separation between consecutive ranks (no accidental
    # ties from rounding at the tail, where raw already floors to 1)
    for i in range(1, n):
        if freqs[i] >= freqs[i - 1]:
            freqs[i] = max(1, freqs[i - 1] - 1)
    return freqs  # freqs[0] = largest ("rank 1"), freqs[-1] = smallest


def arrange_arrival(freqs_by_rank, pattern, rng):
    """freqs_by_rank[0] = largest .. freqs_by_rank[-1] = smallest.
    Returns f[] indexed by ARRIVAL position (arrival index i -> f_i)."""
    n = len(freqs_by_rank)
    if pattern == "desc":
        # largest arrives first
        return list(freqs_by_rank)
    if pattern == "asc":
        # smallest arrives first, largest arrives LAST (classic trap)
        return list(reversed(freqs_by_rank))
    if pattern == "random":
        arr = list(freqs_by_rank)
        rng.shuffle(arr)
        return arr
    if pattern == "cluster":
        # first ~70% of arrivals: the SMALL-rank symbols (bottom of the
        # distribution), locally shuffled so short-window "so far" stats
        # look flat/misleading; final ~30%: the big-rank ("whale") symbols
        # arriving in a late burst.
        k = max(1, int(round(n * 0.7)))
        small_pool = list(freqs_by_rank[n - k:])   # smallest k
        big_pool = list(freqs_by_rank[:n - k])      # remaining (largest) go last
        rng.shuffle(small_pool)
        rng.shuffle(big_pool)
        return small_pool + big_pool
    if pattern == "asc_noisy":
        arr = list(reversed(freqs_by_rank))
        # a handful of local transpositions so it isn't perfectly monotone
        n_swaps = max(1, n // 12)
        for _ in range(n_swaps):
            i = rng.randrange(n - 1)
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
        return arr
    raise ValueError(pattern)


PLAN = {
    1: (8, "random", 2.4),
    2: (12, "desc", 2.4),
    3: (20, "random", 2.8),
    4: (30, "asc", 2.8),
    5: (45, "random", 3.0),
    6: (60, "cluster", 3.0),
    7: (85, "desc", 3.2),
    8: (110, "asc", 3.2),
    9: (150, "cluster", 3.4),
    10: (200, "asc_noisy", 3.4),
}


def main():
    test_id = int(sys.argv[1])
    rng = random.Random(1_000_003 * test_id + 7)

    n, pattern, alpha = PLAN.get(test_id, (min(200, 8 + 10 * test_id), "random", 1.8 + 0.05 * test_id))

    freqs_by_rank = gen_frequencies(n, rng, alpha)
    f = arrange_arrival(freqs_by_rank, pattern, rng)

    depths = build_tree_shape(n, rng)
    rng.shuffle(depths)  # presentation order of the slot multiset is arbitrary

    out = []
    out.append(str(n))
    out.append(" ".join(str(x) for x in f))
    out.append(" ".join(str(x) for x in depths))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
