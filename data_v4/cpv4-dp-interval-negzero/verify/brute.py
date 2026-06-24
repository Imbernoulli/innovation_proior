import sys
from functools import lru_cache

# Independent brute force.
#
# A "state" is an ordered list of cluster charges (each cluster is a contiguous
# run of the original crystals, charge = sum of that run). A move picks two
# ADJACENT clusters and fuses them: the new charge is their sum, and the reward
# gained equals (left charge) * (right charge). We may stop at any point.
#
# We want the maximum total reward reachable from the initial configuration
# (each crystal its own cluster) by any sequence of fusions, stopping whenever.
#
# Brute force: explore all reachable states. From a tuple of cluster charges,
# every adjacent pair can be fused; recurse and take the max reward over all
# sequences (including the empty sequence -> reward 0). Memoize on the tuple of
# charges PLUS the structure is irrelevant because reward only depends on the
# charges fused, and fusing adjacent clusters of a contiguous layout keeps it a
# contiguous layout; the charge tuple fully determines future options/rewards.

def solve(cs):
    cs = tuple(cs)

    @lru_cache(maxsize=None)
    def rec(state):
        # best additional reward obtainable from `state` (a tuple of charges)
        best = 0  # stop now
        m = len(state)
        for i in range(m - 1):
            gain = state[i] * state[i + 1]
            nxt = state[:i] + (state[i] + state[i + 1],) + state[i + 2:]
            cand = gain + rec(nxt)
            if cand > best:
                best = cand
        return best

    return rec(cs)


def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    cs = [int(data[idx + i]) for i in range(n)]
    print(solve(cs))


if __name__ == "__main__":
    main()
