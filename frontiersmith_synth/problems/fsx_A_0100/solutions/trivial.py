# TIER: trivial
# Dig only the single best cell as a size-1 trench.  This reproduces the
# evaluator's weak reference (best single cell, perimeter 4), so it scores ~0.1
# on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
grid = inst["grid"]
lam = inst["lam"]

best = None
br = bc = 0
for r in range(N):
    for c in range(N):
        s = grid[r][c] - 4 * lam
        if best is None or s > best:
            best = s
            br, bc = r, c

print(json.dumps({"cells": [[br, bc]]}))
