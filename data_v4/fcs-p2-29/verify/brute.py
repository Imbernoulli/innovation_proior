#!/usr/bin/env python3
"""
Independent brute-force oracle for:
  Maximum sum of a contiguous (non-empty) subarray a[l..r], where we may delete
  AT MOST one element from that subarray. The result after deletion must be
  non-empty (so a length-1 subarray cannot have its single element deleted).

We enumerate every subarray [l, r], and for each:
  - candidate with no deletion: sum(a[l..r])
  - candidate deleting one index k in [l..r], but only if the subarray has
    length >= 2 (so something remains): sum(a[l..r]) - a[k]
We take the global maximum.

Reads stdin in the same format as sol.cpp: first token n, then n integers.
"""
import sys


def solve(n, a):
    NEG = float("-inf")
    best = NEG
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            length = r - l + 1
            # no deletion
            if s > best:
                best = s
            # delete exactly one element, only if length >= 2
            if length >= 2:
                # delete the index that is most negative -> equivalently subtract
                # min over a[l..r]; but to be fully independent we enumerate.
                for k in range(l, r + 1):
                    cand = s - a[k]
                    if cand > best:
                        best = cand
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    print(solve(n, a))


if __name__ == "__main__":
    main()
