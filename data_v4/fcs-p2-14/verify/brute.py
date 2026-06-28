#!/usr/bin/env python3
"""Independent brute-force oracle for the CIRCULAR max non-adjacent sum problem.

Positions 0..n-1 are arranged on a cycle: i and i+1 are adjacent for all i, and
position n-1 is adjacent to position 0 (the wrap edge). Choose a subset with no
two cyclically-adjacent positions, maximize the sum. Empty subset allowed, so the
answer is always >= 0.

This oracle enumerates ALL 2^n subsets and checks each for cyclic independence.
Deliberately naive and unrelated to the DP in sol.cpp; intended for small n.
"""
import sys


def solve(n, a):
    best = 0  # empty subset always allowed
    for mask in range(1 << n):
        ok = True
        # check adjacency for the linear edges and the wrap edge
        for i in range(n):
            j = (i + 1) % n
            if (mask >> i) & 1 and (mask >> j) & 1:
                ok = False
                break
        if not ok:
            continue
        s = 0
        for i in range(n):
            if (mask >> i) & 1:
                s += a[i]
        if s > best:
            best = s
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    # Special-case n in {1,2} consistently with the cycle definition.
    # For n == 1 there is no real self-loop; the single element may be taken.
    if n == 1:
        print(max(a[0], 0))
        return
    print(solve(n, a))


if __name__ == "__main__":
    main()
