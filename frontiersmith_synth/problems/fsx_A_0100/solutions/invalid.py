# TIER: invalid
# Greedily grab the best cell, then the best OTHER cell that lies in a different
# hotspot (Manhattan distance >= 4 from the first).  The two cells are never
# 4-adjacent, so the trench is disconnected -> infeasible -> the evaluator scores
# this 0.0.  A classic "collect the treasure, ignore that it must be one pit" bug.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
grid = inst["grid"]

cells = [(grid[r][c], r, c) for r in range(N) for c in range(N)]
cells.sort(reverse=True)
(_, r0, c0) = cells[0]
r1, c1 = r0, c0
for (_, r, c) in cells[1:]:
    if abs(r - r0) + abs(c - c0) >= 4:
        r1, c1 = r, c
        break

print(json.dumps({"cells": [[r0, c0], [r1, c1]]}))
