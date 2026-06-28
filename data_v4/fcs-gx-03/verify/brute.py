#!/usr/bin/env python3
"""Independent brute-force oracle for fcs-gx-03 (max-min placement).

Given n positions and an integer k, choose a sub-multiset of size k of the
positions (one item per chosen slot) so that the minimum pairwise distance
between chosen positions is as large as possible; output that maximum.

This oracle does NOT binary-search. It enumerates every size-k subset of the
n slots, computes the min adjacent gap (positions sorted -> min pairwise gap
is the min adjacent gap), and takes the max over all subsets. Exponential but
obviously correct, used only for small n.
"""
import sys
from itertools import combinations


def solve(n, k, pos):
    if k <= 1:
        return 0
    if k > n:
        return None  # infeasible; the generator never emits this case
    pos = sorted(pos)
    best = -1
    # choose k of the n slots (by index, so duplicate positions are distinct slots)
    for combo in combinations(range(n), k):
        vals = sorted(pos[i] for i in combo)
        mn = min(vals[j + 1] - vals[j] for j in range(len(vals) - 1))
        if mn > best:
            best = mn
    return best


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    pos = [int(data[idx + i]) for i in range(n)]
    print(solve(n, k, pos))


if __name__ == "__main__":
    main()
