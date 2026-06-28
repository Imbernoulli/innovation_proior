#!/usr/bin/env python3
"""Demonstrate that 'cheapest-available' greedy is wrong on the seed-0 instance,
and that the DP/brute optimum is strictly better."""
from itertools import permutations

cost = [[1, 1, 100],
        [1, 100, 1],
        [100, 1, 1]]
n = 3

# Greedy: process workers in order, each grabs cheapest still-free task.
used = set()
g = 0
g_assign = []
for i in range(n):
    bestj, bestc = None, None
    for j in range(n):
        if j in used:
            continue
        if bestc is None or cost[i][j] < bestc:
            bestc, bestj = cost[i][j], j
    used.add(bestj)
    g_assign.append(bestj)
    g += bestc
print("greedy assignment:", g_assign, "cost:", g)

# Optimal by exhaustive permutation.
best, bestp = None, None
for p in permutations(range(n)):
    t = sum(cost[i][p[i]] for i in range(n))
    if best is None or t < best:
        best, bestp = t, p
print("optimal assignment:", list(bestp), "cost:", best)
assert g > best, "greedy should be strictly worse here"
print("Greedy is strictly worse:", g, ">", best)
