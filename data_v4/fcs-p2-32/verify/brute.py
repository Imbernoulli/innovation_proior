#!/usr/bin/env python3
"""Independent brute oracle for max dot product of equal-length non-empty subsequences.

Reads the same stdin format as sol.cpp:
    n m
    A[0..n-1]
    B[0..m-1]
Enumerates every pair of non-empty subsequences (chosen index sets, kept in
increasing order) of equal length, computes the dot product, prints the max.
Exponential -- only valid for tiny n, m.
"""
import sys
from itertools import combinations


def solve(n, m, A, B):
    best = None
    # k = common length, from 1 up to min(n, m)
    for k in range(1, min(n, m) + 1):
        for ia in combinations(range(n), k):
            sa = [A[t] for t in ia]
            for ib in combinations(range(m), k):
                dot = 0
                for t in range(k):
                    dot += sa[t] * B[ib[t]]
                if best is None or dot > best:
                    best = dot
    return best


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    A = [int(data[idx + t]) for t in range(n)]; idx += n
    B = [int(data[idx + t]) for t in range(m)]; idx += m
    print(solve(n, m, A, B))


if __name__ == "__main__":
    main()
