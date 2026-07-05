# TIER: invalid
# Fence two separate compounds: the top-left and bottom-right corners.  On every
# instance in this family (grids are at least 12x12) these two cells are NOT
# 4-connected, so the cordon is two components -> the evaluator rejects it as
# infeasible and scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
H, W = inst["H"], inst["W"]

print(json.dumps({"cells": [[0, 0], [H - 1, W - 1]]}))
