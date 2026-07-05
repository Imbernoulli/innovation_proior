# TIER: trivial
# Single best cell: stake the whole base on the one grid cell with the highest
# terrain value.  This reproduces the evaluator's weak baseline (best single-cell
# footprint), so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
H = inst["H"]
W = inst["W"]
grid = inst["grid"]

best = None
br = bc = 0
for r in range(H):
    for c in range(W):
        if best is None or grid[r][c] > best:
            best = grid[r][c]
            br, bc = r, c

print(json.dumps({"cells": [[br, bc]]}))
