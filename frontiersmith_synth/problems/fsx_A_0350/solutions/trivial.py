# TIER: trivial
# Single best tile.  Occupy exactly the one tile with the highest net value.  A
# one-tile footprint is always 4-connected and within budget, so it is valid -- but it
# gathers no coverage, reproducing the evaluator's weak baseline for a score of ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; net = inst["net"]

best = [0, 0]; bv = net[0][0]
for r in range(n):
    for c in range(n):
        if net[r][c] > bv:
            bv = net[r][c]; best = [r, c]

print(json.dumps({"cells": [best]}))
