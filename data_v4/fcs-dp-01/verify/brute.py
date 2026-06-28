#!/usr/bin/env python3
"""
Brute / oracle for "Aliens-Trick Job Split" (max value of EXACTLY k disjoint
non-empty contiguous segments of an integer array).

Straightforward O(n*k) per-k dynamic programming, no Lagrangian trick.
Obviously correct, slow; used as the differential-testing oracle on small n.

Input  (stdin):  n k
                 a_0 a_1 ... a_{n-1}
Output (stdout): a single integer: the maximum total value over all ways to
                 choose exactly k disjoint non-empty contiguous segments.
                 The statement guarantees 1 <= k <= (n+1)//2, so a valid
                 selection always exists.
"""
import sys


def solve(n, k, a):
    NEG = float("-inf")
    # Process elements one at a time, keeping two layers indexed by j (# of
    # segments touched so far):
    #   closed[j] = best value, exactly j segments fully closed, current
    #               element NOT inside an open segment.
    #   open_[j]  = best value, exactly j segments where the j-th one is still
    #               open and includes the last processed element.
    closed = [NEG] * (k + 1)
    open_ = [NEG] * (k + 1)
    closed[0] = 0  # empty prefix: zero segments, nothing open

    for x in a:
        new_closed = [NEG] * (k + 1)
        new_open = [NEG] * (k + 1)
        for j in range(k + 1):
            # current element NOT inside an open segment:
            best = closed[j]
            if open_[j] != NEG:          # close a previously open j-th segment
                best = max(best, open_[j])
            new_closed[j] = best
            # current element IS inside the open j-th segment:
            cand = NEG
            if open_[j] != NEG:          # extend the open j-th segment
                cand = max(cand, open_[j] + x)
            if j >= 1 and closed[j - 1] != NEG:   # start j-th after a >=1 gap
                cand = max(cand, closed[j - 1] + x)
            new_open[j] = cand
        closed = new_closed
        open_ = new_open

    ans = closed[k]
    if open_[k] != NEG:
        ans = max(ans, open_[k])
    return ans


def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    print(solve(n, k, a))


if __name__ == "__main__":
    main()
