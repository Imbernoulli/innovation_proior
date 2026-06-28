#!/usr/bin/env python3
"""Independent brute-force oracle for the equal-sum partition problem.

Reads stdin: n, then n integers.
Decides whether the multiset can be split into two subsets with equal sum.
Prints YES or NO.

Two independent methods, cross-checked against each other when both are
affordable:
  - exhaustive 2^n subset enumeration (used when n is tiny)
  - meet-in-the-middle (used otherwise; still exact, no DP over sum)
This deliberately avoids the subset-sum DP that sol.cpp uses, so the oracle
is algorithmically independent.
"""
import sys


def can_partition_exhaustive(a):
    total = sum(a)
    if total % 2 != 0:
        return False
    half = total // 2
    n = len(a)
    # enumerate all subset sums via bitmask
    for mask in range(1 << n):
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += a[idx]
            m >>= 1
            idx += 1
        if s == half:
            return True
    return False


def can_partition_mitm(a):
    total = sum(a)
    if total % 2 != 0:
        return False
    half = total // 2
    n = len(a)
    mid = n // 2
    left = a[:mid]
    right = a[mid:]

    def subset_sums(arr):
        sums = {0}
        for v in arr:
            sums |= {s + v for s in sums}
        return sums

    ls = subset_sums(left)
    rs = subset_sums(right)
    rs_set = set(rs)
    for s in ls:
        if (half - s) in rs_set:
            return True
    return False


def solve(a):
    n = len(a)
    if n <= 20:
        res = can_partition_exhaustive(a)
        # cross-check with mitm for confidence on small inputs
        assert res == can_partition_mitm(a), "internal oracle disagreement"
        return res
    else:
        return can_partition_mitm(a)


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    print("YES" if solve(a) else "NO")


if __name__ == "__main__":
    main()
