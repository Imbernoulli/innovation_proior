# TIER: greedy
"""The obvious first attempt: the additive-substituent-effects model taken
literally. Score every library substituent by its own additive contribution
(electronic term + a per-item steric bonus term, exactly as the SEARCH
heuristic in the innovation hook), rank by value-per-synthesis-step density,
and pack the single best-density substituent into as many ring positions as
the budget allows (then the next-best type once the top one no longer
fits), filling positions left to right. This is unbounded-knapsack-by-
density -- the textbook first move -- and it has NO notion of ring
adjacency at all, so whenever the best-density substituent is also bulky it
gets packed onto CONSECUTIVE ring positions, which is exactly where the
steric-clash correction (two adjacent bulky groups) detonates and flips
that pair's net contribution negative."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    N = int(nxt())
    K = int(nxt())
    budget = int(nxt())
    P0 = float(nxt())
    alpha = float(nxt())
    beta = float(nxt())
    s_thresh = float(nxt())
    target = float(nxt())
    window = float(nxt())
    lib = []
    for _ in range(K):
        e = float(nxt())
        s = float(nxt())
        c = int(nxt())
        lib.append((e, s, c))

    def value(t):
        e, s, c = lib[t]
        return e + alpha * s

    types = list(range(K))
    types.sort(key=lambda t: (-(value(t) / lib[t][2]), t))

    assign = [0] * N  # 0 = H / empty
    remaining = budget
    p = 0
    while p < N:
        placed = False
        for t in types:
            v = value(t)
            c = lib[t][2]
            if v > 0 and c <= remaining:
                assign[p] = t + 1  # output uses 1-indexed substituent ids
                remaining -= c
                placed = True
                break
        p += 1
        if not placed:
            continue

    print(" ".join(str(a) for a in assign))


if __name__ == "__main__":
    main()
