#!/usr/bin/env python3
"""Independent brute oracle for the minimum-cost assignment problem.

Reads the same stdin format as sol.cpp:
    n
    then n*n integers, row i = costs of worker i for tasks 0..n-1.

Computes the minimum total cost over ALL permutations (worker i -> task perm[i]),
by exhaustive enumeration. This is a completely different method from the bitmask
DP (it enumerates permutations directly), so it is a valid independent check.
Only feasible for small n (<= ~9), which is what the generator produces.
"""
import sys
from itertools import permutations


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    cost = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(data[idx])); idx += 1
        cost.append(row)

    if n == 0:
        print(0)
        return

    best = None
    for perm in permutations(range(n)):
        total = 0
        for i in range(n):
            total += cost[i][perm[i]]
        if best is None or total < best:
            best = total
    print(best)


if __name__ == "__main__":
    main()
