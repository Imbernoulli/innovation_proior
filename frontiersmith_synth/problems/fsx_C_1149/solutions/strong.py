# TIER: strong
"""
The insight: a cut's touch cost is a PURE FUNCTION of its own position,
pen(b) = sum of w_i over passes straddling b, independent of any other cut.
So the whole problem is "changepoint matching": pick a subset of the passes'
own endpoints (the only positions worth cutting at -- cutting anywhere else
only pays build cost with zero chance of touch benefit) to minimize

    sum of pen(chosen cuts)  +  sum of BASE + size^1.5 over resulting slabs

This is a textbook 1-D segmentation DP over the (coordinate-compressed)
candidate set, and it automatically recovers BOTH the widely spaced coarse
cuts and the extra local cuts needed inside a tightly clustered region --
no single fixed granularity has to be chosen in advance, because each
candidate is judged on its own merit within the DP.
"""
import sys
import bisect

GAMMA = 1.5


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    L = int(next(it)); Q = int(next(it)); BASE = int(next(it))
    for _ in range(L):
        next(it)  # hardness profile: not part of the true signal
    queries = []
    for _ in range(Q):
        l = int(next(it)); r = int(next(it)); w = int(next(it))
        queries.append((l, r, w))

    cand = sorted(set(v for (l, r, w) in queries for v in (l, r) if 0 < v < L))

    # coordinate list including sentinels 0 and L
    coords = [0] + cand + [L]

    # pen(c) for each interior candidate
    pen = [0.0] * len(coords)
    for (l, r, w) in queries:
        # candidates strictly inside (l, r): binary-search the coords array
        lo = bisect.bisect_right(coords, l)
        hi = bisect.bisect_left(coords, r)
        for j in range(lo, hi):
            pen[j] += w

    n = len(coords)
    INF = float("inf")
    dp = [INF] * n
    par = [-1] * n
    dp[0] = 0.0
    for j in range(1, n):
        best = INF
        bi = -1
        cj = coords[j]
        penj = pen[j] if j < n - 1 else 0.0  # sentinel L pays no pen
        for i in range(j):
            size = cj - coords[i]
            cost = dp[i] + BASE + size ** GAMMA + penj
            if cost < best:
                best = cost
                bi = i
        dp[j] = best
        par[j] = bi

    # reconstruct
    chosen = []
    j = n - 1
    while j > 0:
        i = par[j]
        if i > 0:
            chosen.append(coords[i])
        j = i
    chosen.reverse()

    print(len(chosen))
    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
