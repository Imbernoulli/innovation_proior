# TIER: trivial
# Do the least a coder could do and still satisfy each ROW's Latin property: for every
# destroyed cell, list the symbols missing from that row (in increasing order) and drop
# them into the empty cells left-to-right. No column awareness, no group reasoning at
# all -- pure per-row bookkeeping. This keeps every row a valid permutation (so it does
# not get shredded by its own row-duplicate penalty) but the arbitrary left-to-right
# order it picks has no reason to match the true hidden table, so correctness on
# destroyed cells is low and column-duplicate violations are common.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
grid = [row[:] for row in inst["grid"]]

for i in range(n):
    known = set(x for x in grid[i] if x is not None)
    missing = sorted(set(range(n)) - known)
    k = 0
    for j in range(n):
        if grid[i][j] is None:
            grid[i][j] = missing[k]
            k += 1

print(json.dumps({"grid": grid}))
