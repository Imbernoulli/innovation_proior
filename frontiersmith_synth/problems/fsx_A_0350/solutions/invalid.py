# TIER: invalid
# Two DISCONNECTED tiles: occupy the two opposite corners of the grid.  For N >= 2
# these are not 4-adjacent, so the footprint fails the connectivity check and the
# evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

print(json.dumps({"cells": [[0, 0], [n - 1, n - 1]]}))
