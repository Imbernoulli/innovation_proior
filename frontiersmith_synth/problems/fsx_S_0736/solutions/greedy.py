# TIER: greedy
"""
The textbook per-class extractor: process every stall bottom-up (indices are already
topologically sorted -- every child index is strictly greater than its parent) and pick,
independently for EACH stall, the candidate minimizing (own_cost + sum of children's own
best-cost).  This is exactly the standard recursive/tree-cost e-graph extraction algorithm.
It ignores that a downstream stall might be reused by many parents (sharing) -- each stall
reasons only about its own subtree in isolation.
"""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v

    N = int(nxt()); R = int(nxt())
    roots = [int(nxt()) for _ in range(R)]
    classes = []
    for _i in range(N):
        M = int(nxt())
        cands = []
        for _k in range(M):
            cost = int(nxt())
            L = int(nxt())
            children = [int(nxt()) for _ in range(L)]
            cands.append((cost, children))
        classes.append(cands)

    best = [0] * N          # best per-stall TREE cost (double counts shared subtrees)
    sel = [0] * N
    for i in range(N - 1, -1, -1):
        b = None
        bk = 0
        for k, (cost, children) in enumerate(classes[i]):
            c = cost
            for ch in children:
                c += best[ch]
            if b is None or c < b:
                b = c; bk = k
        best[i] = b
        sel[i] = bk

    sys.stdout.write("\n".join(str(x) for x in sel) + "\n")


if __name__ == "__main__":
    main()
