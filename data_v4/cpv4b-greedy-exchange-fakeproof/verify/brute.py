#!/usr/bin/env python3
"""Independent brute force for the gondola problem.

Minimum number of gondola trips, each carrying at most 2 climbers whose
combined weight is at most C. This equals n minus the maximum number of
valid (sum <= C) pairs we can form, which we find by exhaustive search over
matchings (correct but exponential -- oracle only).
"""
import sys


def solve(w, C):
    n = len(w)
    best = [0]  # max number of valid pairs
    used = [False] * n

    def rec(pairs):
        if pairs > best[0]:
            best[0] = pairs
        i = 0
        while i < n and used[i]:
            i += 1
        if i == n:
            return
        used[i] = True
        # leave i unpaired (it takes a solo trip)
        rec(pairs)
        # try pairing i with every later free climber that fits
        for j in range(i + 1, n):
            if not used[j] and w[i] + w[j] <= C:
                used[j] = True
                rec(pairs + 1)
                used[j] = False
        used[i] = False

    rec(0)
    return n - best[0]


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    w = [int(data[idx + k]) for k in range(n)]
    print(solve(w, C))


if __name__ == "__main__":
    main()
