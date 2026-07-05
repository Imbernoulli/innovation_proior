# TIER: greedy
# Single-seed region growing.  Start the trench at the single best cell, then
# repeatedly add whichever boundary neighbour most increases the plan value
# (yield gained minus the change in shoring perimeter).  Stop at the first local
# optimum where no single addition helps.  This captures roughly one hotspot but
# never reconsiders / removes cells and never tries other seeds.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
grid = inst["grid"]
lam = inst["lam"]

NEI = ((1, 0), (-1, 0), (0, 1), (0, -1))


def seed_best():
    best = None
    br = bc = 0
    for r in range(N):
        for c in range(N):
            s = grid[r][c] - 4 * lam
            if best is None or s > best:
                best = s
                br, bc = r, c
    return br, bc


def delta_add(dug, r, c):
    # change in plan value from adding cell (r,c) that is adjacent to `dug`.
    dv = grid[r][c]
    dp = 0
    for dr, dc in NEI:
        nb = (r + dr, c + dc)
        if nb in dug:
            dp -= 1          # this face was perimeter, now internal
        else:
            dp += 1          # new exposed face
    return dv - lam * dp


br, bc = seed_best()
dug = {(br, bc)}
frontier = set()
for dr, dc in NEI:
    nb = (br + dr, bc + dc)
    if 0 <= nb[0] < N and 0 <= nb[1] < N:
        frontier.add(nb)

while True:
    best_gain = 0
    best_cell = None
    for (r, c) in frontier:
        g = delta_add(dug, r, c)
        if g > best_gain:
            best_gain = g
            best_cell = (r, c)
    if best_cell is None:
        break
    dug.add(best_cell)
    frontier.discard(best_cell)
    r, c = best_cell
    for dr, dc in NEI:
        nb = (r + dr, c + dc)
        if 0 <= nb[0] < N and 0 <= nb[1] < N and nb not in dug:
            frontier.add(nb)

print(json.dumps({"cells": [[r, c] for (r, c) in dug]}))
