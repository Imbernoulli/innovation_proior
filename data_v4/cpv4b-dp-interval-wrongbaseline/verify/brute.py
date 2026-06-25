import sys
from functools import lru_cache

# INDEPENDENT brute force.
# We have n reels arranged in a circle. A "reel" has a length (weight). At each
# step we may splice two ADJACENT reels (adjacent on the current circle, which
# includes the wraparound between the first and last) into one reel whose length
# is the sum; the cost of that splice equals the sum of the two spliced lengths.
# We repeat until one reel remains. Minimize total splicing cost.
#
# This brute force literally simulates the circle as a tuple of reel lengths and
# tries every adjacent merge at every step (including the wraparound pair when
# there are >= 3 reels; with exactly 2 reels there is only one pair). It memoizes
# on the multiset-as-tuple state -- but rotations matter for adjacency, so we keep
# the tuple order and let the recursion explore. To keep it correct we DO NOT
# collapse rotations; we just brute every adjacent pair.

def solve(weights):
    n = len(weights)
    if n <= 1:
        return 0

    seen = {}

    def rec(circle):
        # circle is a tuple of current reel lengths in circular order.
        c = len(circle)
        if c == 1:
            return 0
        if circle in seen:
            return seen[circle]
        best = None
        # adjacent pairs: (i, i+1) for i in 0..c-1, where index c wraps to 0.
        # When c == 2 the pair (0,1) and the wrap pair (1,0) are the same merge,
        # so only consider i in range(c) but skip the duplicate wrap when c == 2.
        pairs = []
        for i in range(c):
            j = (i + 1) % c
            if c == 2 and i == 1:
                continue  # avoid duplicating the single pair
            pairs.append((i, j))
        for (i, j) in pairs:
            merged = circle[i] + circle[j]
            cost = merged
            # build new circle by replacing the two adjacent reels with one.
            if j == (i + 1):  # normal adjacency, no wrap
                newc = circle[:i] + (merged,) + circle[i + 2:]
            else:  # wrap: i == c-1, j == 0
                newc = (merged,) + circle[1:c - 1]
            val = cost + rec(newc)
            if best is None or val < best:
                best = val
        seen[circle] = best
        return best

    return rec(tuple(weights))


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    w = [int(data[idx + i]) for i in range(n)]
    print(solve(w))


if __name__ == "__main__":
    main()
