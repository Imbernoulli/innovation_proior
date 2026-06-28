#!/usr/bin/env python3
"""
Independent brute-force oracle for the Longest Bitonic (strictly increase-then-decrease)
Subsequence problem.

A subsequence b[0..k-1] (taken in original order) is BITONIC iff there is a peak position
p (0 <= p <= k-1) such that:
    b[0] < b[1] < ... < b[p]      (strictly increasing up to the peak)
    b[p] > b[p+1] > ... > b[k-1]  (strictly decreasing after the peak)
AND there is at least one real increase, i.e. p >= 1 (the array must go up before it comes
down). The decreasing tail may be empty (so the minimum valid bitonic length is 2).

INDEPENDENT METHOD (different from the shipped LIS-from-both-sides combine):
We search the space of bitonic subsequences directly with a memoized DFS over states
(last chosen index, phase), where phase 0 = still allowed to go up or switch to down,
phase 1 = committed to going down. We start the chain at every index in the "up" phase,
force at least one up-step before any down-step (tracked by a `went_up` flag), and take
the maximum length over all valid chains. This is a state-space exploration, structurally
unrelated to computing inc[i]/dec[i] arrays, so it serves as a genuine cross-check.
"""
import sys
sys.setrecursionlimit(1 << 25)


def solve(a):
    n = len(a)
    if n == 0:
        return 0

    from functools import lru_cache

    # f(i, phase, went_up): best ADDITIONAL count of elements we can still append AFTER
    # having chosen index i, where phase indicates whether we are still ascending (0) or
    # have begun descending (1). went_up records whether at least one strict increase has
    # occurred among the chosen elements so far (including the step that led to i).
    # Returns total length of the bitonic chain from i onward (counting i itself) that is
    # VALID, or -infinity if no valid completion exists from this state.
    NEG = -1 << 30

    @lru_cache(maxsize=None)
    def f(i, phase, went_up):
        best = 1 + (1000000000 if False else 0)  # length counting i itself
        # A standalone i with no down-tail is only valid if we already went up.
        valid_here = (went_up == 1)
        res = (1 if valid_here else NEG)
        # try to extend
        if phase == 0:
            # we may go up (stay phase 0) to a strictly larger value
            for j in range(i + 1, n):
                if a[j] > a[i]:
                    cand = 1 + f(j, 0, 1)
                    if cand > res:
                        res = cand
            # or we may go down (switch to phase 1) to a strictly smaller value,
            # but only if we have already gone up at least once
            if went_up == 1:
                for j in range(i + 1, n):
                    if a[j] < a[i]:
                        cand = 1 + f(j, 1, 1)
                        if cand > res:
                            res = cand
        else:  # phase == 1, descending only
            for j in range(i + 1, n):
                if a[j] < a[i]:
                    cand = 1 + f(j, 1, 1)
                    if cand > res:
                        res = cand
        return res

    best = 0
    for i in range(n):
        v = f(i, 0, 0)  # start at i, ascending phase, no increase yet
        if v > best:
            best = v
    f.cache_clear()
    return best


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    print(solve(a))


if __name__ == "__main__":
    main()
