# TIER: trivial
# Fence off ONLY the single most-profitable stack: pick the cell maximizing
# v - 4*lam (a one-cell cordon has perimeter 4).  This exactly reproduces the
# evaluator's weak anchor `base`, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
H, W, lam = inst["H"], inst["W"], inst["lam"]
grid = inst["grid"]

best = None
best_val = None
for i in range(H):
    for j in range(W):
        val = grid[i][j] - 4 * lam
        if best_val is None or val > best_val:
            best_val = val
            best = [i, j]

print(json.dumps({"cells": [best]}))
