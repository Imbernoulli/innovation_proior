# TIER: greedy
"""
The obvious "recipe" move: pick a block count via the generic sqrt-decomposition
reflex (K ~= sqrt(Q), a very standard bucket-count heuristic for range
problems), then place those K-1 cuts by balancing the posted hardness *mass*
-- an intuitively reasonable way to make every slab "comparably hard" to
lift. It never reads BASE (so it has no idea how many slabs the convex
build penalty actually wants) and never looks at the inspection passes'
endpoints (so wherever the hardness profile's shape is decorrelated from
where the passes cluster, it pays an avoidable touch penalty on top).
"""
import sys
import math


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    L = int(next(it)); Q = int(next(it)); _BASE = int(next(it))
    h = [int(next(it)) for _ in range(L)]
    for _ in range(Q):
        next(it); next(it); next(it)  # queries unused by this heuristic

    K = max(1, round(math.sqrt(Q)))
    K = min(K, L - 1)

    if K <= 1:
        print(0)
        print()
        return

    total = sum(h)
    target = total / K
    cuts = []
    prefix = 0
    for x in range(L):
        prefix += h[x]
        if prefix >= target * (len(cuts) + 1) and 0 < x + 1 < L:
            cuts.append(x + 1)
            if len(cuts) == K - 1:
                break

    seen = []
    last = -1
    for c in cuts:
        if c != last and 0 < c < L:
            seen.append(c)
            last = c

    print(len(seen))
    print(" ".join(map(str, seen)))


if __name__ == "__main__":
    main()
