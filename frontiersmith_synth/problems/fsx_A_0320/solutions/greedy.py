# TIER: greedy
# Single-seed region growing.  Start from the best single stack (max v - 4*lam),
# then repeatedly annex the frontier cell whose addition most increases profit,
# stopping the moment no addition helps.  Adding a cell with `a` already-selected
# neighbours changes profit by  v[c] - lam*(4 - 2*a)  (it exposes 4 new edges but
# seals 2*a previously-exposed ones).  Better than a one-cell cordon because it
# absorbs adjacent value, but it never restarts from other seeds or backtracks.
import sys, json

inst = json.load(sys.stdin)
H, W, lam = inst["H"], inst["W"], inst["lam"]
grid = inst["grid"]


def nbrs(i, j):
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W:
            yield ni, nj


# best single seed
seed = None
best_val = None
for i in range(H):
    for j in range(W):
        val = grid[i][j] - 4 * lam
        if best_val is None or val > best_val:
            best_val = val
            seed = (i, j)

S = {seed}
frontier = {}
for n in nbrs(*seed):
    frontier[n] = 1

while frontier:
    best_delta = 0
    best_c = None
    for c, a in frontier.items():
        delta = grid[c[0]][c[1]] - lam * (4 - 2 * a)
        if delta > best_delta:
            best_delta = delta
            best_c = c
    if best_c is None:
        break
    S.add(best_c)
    del frontier[best_c]
    for n in nbrs(*best_c):
        if n not in S:
            frontier[n] = frontier.get(n, 0) + 1

print(json.dumps({"cells": [[i, j] for (i, j) in S]}))
