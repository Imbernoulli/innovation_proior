#!/usr/bin/env python3
from itertools import permutations
cost = [[0, 6, 3],
        [6, 0, 8],
        [3, 7, 7]]
n = 3
# global cheapest-cell greedy
cells = sorted(((cost[i][j], i, j) for i in range(n) for j in range(n)))
uw, ut, tot, assign = set(), set(), 0, {}
for c, i, j in cells:
    if i in uw or j in ut:
        continue
    uw.add(i); ut.add(j); tot += c; assign[i] = j
    if len(uw) == n:
        break
print("global greedy assign:", [assign[i] for i in range(n)], "cost:", tot)
# in-order greedy
ut, tot2, asg2 = set(), 0, []
for i in range(n):
    bj, bc = None, None
    for j in range(n):
        if j in ut:
            continue
        if bc is None or cost[i][j] < bc:
            bc, bj = cost[i][j], j
    ut.add(bj); tot2 += bc; asg2.append(bj)
print("inorder greedy assign:", asg2, "cost:", tot2)
# optimal
best, bp = None, None
for p in permutations(range(n)):
    t = sum(cost[i][p[i]] for i in range(n))
    if best is None or t < best:
        best, bp = t, p
print("optimal assign:", list(bp), "cost:", best)
