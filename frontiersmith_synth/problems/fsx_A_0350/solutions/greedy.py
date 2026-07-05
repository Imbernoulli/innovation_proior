# TIER: greedy
# Single-deposit region growing (no refinement, no bridging).  Seed the footprint at
# the globally best tile, then repeatedly annex the frontier tile (adjacent to the
# current region) with the largest net value, as long as it is strictly positive and
# the budget K is not exhausted.  This vacuums up one whole positive deposit and stops
# at the surrounding net-negative regolith moat.  It clears the single-tile baseline
# easily, but being myopic it never spends budget on negative bridge tiles to reach a
# second rich deposit -- yield the strong tier's multi-seed + bridging recovers.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; k = inst["k"]; net = inst["net"]

best = (0, 0); bv = net[0][0]
for r in range(n):
    for c in range(n):
        if net[r][c] > bv:
            bv = net[r][c]; best = (r, c)

S = {best}
while len(S) < k:
    cand = None; cg = 0
    for (r, c) in S:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in S:
                if net[nr][nc] > cg:
                    cg = net[nr][nc]; cand = (nr, nc)
    if cand is None or cg <= 0:
        break
    S.add(cand)

print(json.dumps({"cells": [[r, c] for (r, c) in S]}))
