#!/usr/bin/env python3
"""Search for a small assignment matrix where the GLOBAL cheapest-cell greedy
(repeatedly pick the cheapest remaining free (worker,task) cell) is strictly
worse than the optimal assignment. Also report whether in-order greedy fails."""
import random
from itertools import permutations


def optimal(cost, n):
    best = None
    for p in permutations(range(n)):
        t = sum(cost[i][p[i]] for i in range(n))
        if best is None or t < best:
            best = t
    return best


def global_greedy(cost, n):
    cells = sorted(((cost[i][j], i, j) for i in range(n) for j in range(n)))
    usedw, usedt = set(), set()
    tot = 0
    cnt = 0
    for c, i, j in cells:
        if i in usedw or j in usedt:
            continue
        usedw.add(i); usedt.add(j); tot += c; cnt += 1
        if cnt == n:
            break
    return tot


def inorder_greedy(cost, n):
    usedt = set()
    tot = 0
    for i in range(n):
        bj, bc = None, None
        for j in range(n):
            if j in usedt:
                continue
            if bc is None or cost[i][j] < bc:
                bc, bj = cost[i][j], j
        usedt.add(bj); tot += bc
    return tot


def main():
    rng = random.Random(1)
    n = 3
    for _ in range(200000):
        cost = [[rng.randint(0, 9) for _ in range(n)] for _ in range(n)]
        opt = optimal(cost, n)
        gg = global_greedy(cost, n)
        ig = inorder_greedy(cost, n)
        if gg > opt and ig > opt:
            print("FOUND (both greedies worse):")
            for row in cost:
                print(row)
            print("optimal:", opt, "global_greedy:", gg, "inorder_greedy:", ig)
            return
    print("none found for both; searching for global-greedy-only failure")
    rng = random.Random(2)
    for _ in range(200000):
        cost = [[rng.randint(0, 9) for _ in range(n)] for _ in range(n)]
        opt = optimal(cost, n)
        gg = global_greedy(cost, n)
        if gg > opt:
            print("FOUND (global greedy worse):")
            for row in cost:
                print(row)
            print("optimal:", opt, "global_greedy:", gg, "inorder_greedy:", inorder_greedy(cost, n))
            return


if __name__ == "__main__":
    main()
