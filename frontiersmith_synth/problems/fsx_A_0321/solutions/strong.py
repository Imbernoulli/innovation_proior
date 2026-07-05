# TIER: strong
# Multi-restart yield-priority greedy.  Beyond the single deterministic pass, run
# several SEEDED restarts whose priority is the water yield perturbed by a small
# deterministic jitter (different tie-breaking / near-tie reorderings), then keep
# the highest-yield valid cap set found.  Exploring several orderings escapes the
# myopia of one greedy pass and finds heavier caps on many instances.
import sys
import random


def str_of(idx, n):
    d = []
    for _ in range(n):
        d.append(idx % 3)
        idx //= 3
    return "".join(str(x) for x in reversed(d))


def idx_of(s):
    v = 0
    for c in s:
        v = v * 3 + (ord(c) - 48)
    return v


def third(x, y):
    return "".join(str((-(int(a) + int(b))) % 3) for a, b in zip(x, y))


def build(order):
    S = set()
    for p in order:
        ok = True
        for a in S:
            if third(p, a) in S:
                ok = False
                break
        if ok:
            S.add(p)
    return S


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    blocked = set(next(it) for _ in range(m))
    N = 3 ** n
    weights = [int(next(it)) for _ in range(N)]

    W = {}
    plots = []
    for i in range(N):
        s = str_of(i, n)
        if s not in blocked:
            W[s] = weights[i]
            plots.append(s)

    # deterministic yield-priority pass
    best = build(sorted(plots, key=lambda s: -W[s]))
    besty = sum(W[p] for p in best)

    # seeded restarts with jittered priority (fewer at large n for the time budget)
    K = 12 if n <= 5 else (8 if n <= 7 else 3)
    for r in range(K):
        rng = random.Random(31 * r + 7 * n + 1)
        scale = rng.choice([50, 120, 250, 400])
        order = sorted(plots, key=lambda s: -(W[s] + rng.uniform(-scale, scale)))
        S = build(order)
        y = sum(W[p] for p in S)
        if y > besty:
            besty = y
            best = S
    sys.stdout.write("\n".join(best) + "\n")


main()
