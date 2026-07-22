# TIER: trivial
"""Naive first-fit lexicographic edge coloring: process pairs (1,2),(1,3),...
in order, place each on the earliest legal (date,court) slot. Ignores the
weather scenarios entirely and has no notion of round-robin structure."""
import sys


def canonical_pairs(N):
    return [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]


def first_fit_schedule(N, D, C):
    busy = [set() for _ in range(D + 1)]
    ccount = [0] * (D + 1)
    sched = []
    for (i, j) in canonical_pairs(N):
        for d in range(1, D + 1):
            if ccount[d] < C and i not in busy[d] and j not in busy[d]:
                ccount[d] += 1
                busy[d].add(i)
                busy[d].add(j)
                sched.append((d, ccount[d]))
                break
    return sched


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); D = int(next(it)); C = int(next(it))
    K = int(next(it)); next(it); next(it)
    for _ in range(K):
        b = int(next(it))
        for _ in range(b):
            next(it)
    sched = first_fit_schedule(N, D, C)
    out = []
    for d, c in sched:
        out.append(f"{d} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
