#!/usr/bin/env python3
# Independent brute-force oracle for "Maximum-sum strictly increasing subsequence".
# Exhaustively enumerate ALL 2^n subsequences, keep those strictly increasing in
# value, and report the maximum sum (empty subsequence -> 0). O(2^n * n), only for
# small n. No DP, no greedy: a different algorithm entirely from sol.cpp.
import sys


def solve(a):
    n = len(a)
    best = 0  # empty subsequence allowed
    for mask in range(1 << n):
        idx = [i for i in range(n) if (mask >> i) & 1]
        # idx is already in increasing index order (preserves array order)
        ok = True
        for k in range(1, len(idx)):
            if a[idx[k - 1]] >= a[idx[k]]:  # must be STRICTLY increasing
                ok = False
                break
        if ok:
            s = sum(a[i] for i in idx)
            if s > best:
                best = s
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    print(solve(a))


if __name__ == "__main__":
    main()
