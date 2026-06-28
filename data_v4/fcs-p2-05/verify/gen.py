#!/usr/bin/env python3
"""Random + edge-case generator for the minimum-cost assignment problem.

Usage: gen.py <seed>
Emits a valid test case to stdout in the format sol.cpp expects:
    n
    n rows of n integers (cost[i][j]).

Small n (so the permutation brute oracle stays fast). Includes specially
crafted instances where the cheapest-available greedy is wrong, plus edge
cases (n=0, n=1, ties, negatives, large values).
"""
import random
import sys


def emit(n, cost):
    out = [str(n)]
    for i in range(n):
        out.append(" ".join(str(x) for x in cost[i]))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # ~6% of cases are dedicated edge cases.
    r = rng.random()

    if seed == 0:
        # The greedy-killer counterexample (matches the one in reasoning.md):
        # both cheapest-cell greedy and in-order greedy take the diagonal
        # (0 + 0 + 7 = 7), but the optimum is the cost-6 assignment 0->2,1->1,2->0.
        emit(3, [[0, 6, 3],
                 [6, 0, 8],
                 [3, 7, 7]])
        return
    if seed == 1:
        emit(0, [])
        return
    if seed == 2:
        emit(1, [[rng.randint(-10**6, 10**6)]])
        return

    if r < 0.06:
        # Edge: n=1
        emit(1, [[rng.randint(-10**9, 10**9)]])
        return
    if r < 0.12:
        # Edge: all-equal matrix (every assignment same cost; greedy looks fine).
        n = rng.randint(1, 8)
        v = rng.randint(-10**4, 10**4)
        emit(n, [[v] * n for _ in range(n)])
        return
    if r < 0.18:
        # Edge: negatives allowed.
        n = rng.randint(1, 8)
        emit(n, [[rng.randint(-1000, 1000) for _ in range(n)] for _ in range(n)])
        return
    if r < 0.24:
        # Edge: a forced cheap cell that greedy grabs but globally hurts.
        n = rng.randint(2, 8)
        cost = [[rng.randint(50, 100) for _ in range(n)] for _ in range(n)]
        # Make worker 0's cheapest task very tempting, but it's the only cheap
        # option for worker 1 too.
        cost[0][0] = 1
        cost[1][0] = 2
        for j in range(1, n):
            cost[1][j] = 90  # worker 1 expensive everywhere else
        emit(n, cost)
        return

    # General random case, small n for the brute oracle.
    n = rng.randint(1, 8)
    hi = rng.choice([10, 100, 1000, 10**6])
    lo = -hi if rng.random() < 0.4 else 0
    cost = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]
    emit(n, cost)


if __name__ == "__main__":
    main()
