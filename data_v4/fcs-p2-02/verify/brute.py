#!/usr/bin/env python3
"""Independent brute-force oracle for weighted interval scheduling.

Reads:  n, then n lines of  s e w   (half-open interval [s, e), weight w).
Writes: max total weight of a set of pairwise non-overlapping intervals
        (empty set allowed -> at least 0).

Two independent methods:
  - n <= 18 : exhaustive over all 2^n subsets, checking pairwise compatibility
              directly (no DP at all). Fully model-independent ground truth.
  - n  > 18 : O(n^2) DP that, for each interval, scans ALL earlier intervals
              to find compatible predecessors (no sorting-by-end + binary search).
"""
import sys


def compatible(a, b):
    # non-overlapping iff a ends before/at b starts, or b ends before/at a starts
    return a[1] <= b[0] or b[1] <= a[0]


def solve_exhaustive(jobs):
    n = len(jobs)
    best = 0
    for mask in range(1 << n):
        chosen = [jobs[i] for i in range(n) if (mask >> i) & 1]
        ok = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if not compatible(chosen[i], chosen[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            tot = sum(c[2] for c in chosen)
            if tot > best:
                best = tot
    return best


def solve_dp_quadratic(jobs):
    # Sort by end; dp[i] = best using a chosen set whose LAST interval is i,
    # computed by scanning all earlier-by-end intervals that are compatible.
    js = sorted(jobs, key=lambda t: (t[1], t[0]))
    n = len(js)
    # endsel[i] = best total weight of a compatible chain whose last interval is js[i]
    endsel = [0] * n
    ans = 0
    for i in range(n):
        s_i, e_i, w_i = js[i]
        cur = w_i  # interval i alone
        for j in range(i):
            s_j, e_j, w_j = js[j]
            # j ends no later than i (sorted); j compatible with i iff e_j <= s_i
            if e_j <= s_i:
                if endsel[j] + w_i > cur:
                    cur = endsel[j] + w_i
        endsel[i] = cur
        if cur > ans:
            ans = cur
    return ans


def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    jobs = []
    for _ in range(n):
        s = int(data[idx]); e = int(data[idx + 1]); w = int(data[idx + 2])
        idx += 3
        jobs.append((s, e, w))
    if n <= 18:
        print(solve_exhaustive(jobs))
    else:
        print(solve_dp_quadratic(jobs))


if __name__ == "__main__":
    main()
