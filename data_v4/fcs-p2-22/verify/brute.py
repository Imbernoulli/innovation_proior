#!/usr/bin/env python3
"""Independent brute oracle for Burst Balloons.

Directly simulates the bursting process: at each step pick SOME remaining
balloon to burst next, score it as left*cur*right against the *current*
remaining neighbours (with virtual 1s at the ends), recurse on the rest,
and take the max over all choices. This explores the actual game tree and
makes no use of the 'last balloon to burst' interval-DP idea, so it is an
honest independent check.

We memoize on the tuple of remaining (original-index) balloons to stay
tractable for n up to ~9-11; values themselves don't affect reachability so
the memo key is the set of surviving positions.
"""
import sys
from functools import lru_cache


def solve(nums):
    n = len(nums)
    if n == 0:
        return 0

    @lru_cache(maxsize=None)
    def best(remaining):
        # remaining: sorted tuple of original indices still present
        if not remaining:
            return 0
        res = 0
        for idx, pos in enumerate(remaining):
            left = nums[remaining[idx - 1]] if idx - 1 >= 0 else 1
            right = nums[remaining[idx + 1]] if idx + 1 < len(remaining) else 1
            gain = left * nums[pos] * right
            rest = remaining[:idx] + remaining[idx + 1:]
            cand = gain + best(rest)
            if cand > res:
                res = cand
        return res

    return best(tuple(range(n)))


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    nums = [int(next(it)) for _ in range(n)]
    print(solve(nums))


if __name__ == "__main__":
    main()
