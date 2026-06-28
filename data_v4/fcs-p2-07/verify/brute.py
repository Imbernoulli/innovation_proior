#!/usr/bin/env python3
"""Independent brute oracle for the egg-drop minimum-worst-case-trials problem.

This uses the *floors* DP -- the classic minimax recurrence -- which is a
completely different algorithm from the solution's "cover" recurrence:

    dp[e][f] = minimum number of trials that *guarantees* finding the critical
               floor among f candidate floors using e eggs.

    dp[e][f] = 1 + min_{x=1..f} max( dp[e-1][x-1],   # egg breaks: x-1 floors below
                                     dp[e][f-x] )     # egg survives: f-x floors above
    dp[1][f] = f          (one egg => must scan bottom-up)
    dp[e][0] = 0
    dp[e][1] = 1

The inner minimisation is the plain O(f) scan over the drop position x with no
monotonicity tricks, so the oracle is transparently correct. The generator keeps
m small enough that the resulting O(k*m^2) cost is cheap.
"""
import sys


def solve(k, m):
    if m <= 0:
        return 0
    # Eggs beyond m are never useful (you can never break more than m times),
    # and beyond ~log2(m)+1 they stop helping; cap to keep the table small.
    e_cap = min(k, m)
    # dp[e][f] with rolling rows; prev = dp[e-1], cur = dp[e].
    prev = [f for f in range(m + 1)]  # dp[1][f] = f  (also valid base e=1)
    if e_cap == 1:
        return prev[m]
    for e in range(2, e_cap + 1):
        cur = [0] * (m + 1)
        for f in range(1, m + 1):
            best = None
            for x in range(1, f + 1):
                worst = prev[x - 1]
                other = cur[f - x]
                if other > worst:
                    worst = other
                if best is None or worst < best:
                    best = worst
            cur[f] = 1 + best
        prev = cur
    return prev[m]


def main():
    data = sys.stdin.read().split()
    k = int(data[0])
    m = int(data[1])
    print(solve(k, m))


if __name__ == "__main__":
    main()
