# TIER: invalid
# Claim two far-apart corner cells as the footprint.  They are not adjacent, so
# the set is NOT 4-connected -> the evaluator rejects it as infeasible and scores
# the instance 0.0.
import sys, json

inst = json.load(sys.stdin)
H = inst["H"]
W = inst["W"]

print(json.dumps({"cells": [[0, 0], [H - 1, W - 1]]}))
