# TIER: greedy
# Single-seed region growing.  Start from the highest-value cell and repeatedly
# annex the boundary cell whose addition gives the largest POSITIVE net gain,
# stopping once no boundary cell can improve the footprint.  Adding a cell of
# value v that already touches k chosen neighbours changes the boundary length by
# (4 - 2k), so the gain is  v - perim_cost*(4 - 2k).
# This beats the single-cell baseline by pulling in nearby ice/hotspots, but it
# gets stuck at the first local optimum around one hotspot cluster.
import sys, json

inst = json.load(sys.stdin)
H = inst["H"]
W = inst["W"]
grid = inst["grid"]
pc = inst["perim_cost"]


def neighbours(r, c):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            yield nr, nc


# seed = global best cell
br = bc = 0
best = None
for r in range(H):
    for c in range(W):
        if best is None or grid[r][c] > best:
            best = grid[r][c]
            br, bc = r, c

chosen = {(br, bc)}
# frontier = candidate cells adjacent to the region
frontier = set()
for nb in neighbours(br, bc):
    frontier.add(nb)

while frontier:
    best_gain = 0
    best_cell = None
    for (r, c) in frontier:
        k = sum(1 for nb in neighbours(r, c) if nb in chosen)
        gain = grid[r][c] - pc * (4 - 2 * k)
        if gain > best_gain:
            best_gain = gain
            best_cell = (r, c)
    if best_cell is None:
        break
    chosen.add(best_cell)
    frontier.discard(best_cell)
    for nb in neighbours(*best_cell):
        if nb not in chosen:
            frontier.add(nb)

print(json.dumps({"cells": [[r, c] for (r, c) in chosen]}))
