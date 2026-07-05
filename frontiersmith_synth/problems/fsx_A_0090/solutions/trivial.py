# TIER: trivial
# Overlap-BLIND top-K: drill the K wells at the K hottest cells (tie-break by
# (row, col)).  This is exactly the evaluator's weak baseline, so all K wells pile
# onto the tallest plume, waste capacity by re-tapping the same cells, and the
# instance scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; k = inst["k"]; heat = inst["heat"]

cells = [(heat[r][c], r, c) for r in range(n) for c in range(n)]
cells.sort(key=lambda t: (-t[0], t[1], t[2]))
wells = [[r, c] for (_, r, c) in cells[:k]]

print(json.dumps({"wells": wells}))
