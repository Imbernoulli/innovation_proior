# TIER: strong
# Multi-seed region growing + budget-aware deposit bridging + negative-leaf pruning.
#   1) Grow a single-deposit region (positive-frontier region growing) from each of the
#      top-8 richest seed tiles; keep the best-value single-deposit footprint.
#   2) Bridging: repeatedly try to connect the current footprint to the richest
#      not-yet-included deposit peak via the CHEAPEST path (Dijkstra with edge cost =
#      max(0, -net) of entered tiles, i.e. the negative regolith you must pay to bridge).
#      Adopt the bridge + re-grow only if the total net value strictly increases and the
#      combined footprint still fits the budget K.  This explicitly trades scarce budget
#      on negative bridge tiles for a richer second deposit -- exactly what pure greedy
#      refuses to do.
#   3) Prune any net-negative tile whose removal keeps the region connected.
# Deterministic scan/seed order and a bounded improvement loop keep it reproducible.
# The loose all-positive upper bound keeps even this normalized score below 1.0.
import sys, json, heapq

inst = json.load(sys.stdin)
n = inst["n"]; k = inst["k"]; net = inst["net"]
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def region_grow(S):
    S = set(S)
    while len(S) < k:
        cand = None; cg = 0
        for (r, c) in S:
            for dr, dc in NB:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in S:
                    if net[nr][nc] > cg:
                        cg = net[nr][nc]; cand = (nr, nc)
        if cand is None or cg <= 0:
            break
        S.add(cand)
    return S


def val(S):
    return sum(net[r][c] for (r, c) in S)


def connected(S):
    S = set(S)
    if not S:
        return True
    start = next(iter(S)); stack = [start]; seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NB:
            nb = (r + dr, c + dc)
            if nb in S and nb not in seen:
                seen.add(nb); stack.append(nb)
    return len(seen) == len(S)


tiles = sorted(((net[r][c], r, c) for r in range(n) for c in range(n)), reverse=True)
seeds = [(r, c) for (v, r, c) in tiles[:8] if v > 0]
if not seeds:
    seeds = [(tiles[0][1], tiles[0][2])]

best_S = None; best_v = -10 ** 9
for sd in seeds:
    S = region_grow({sd})
    v = val(S)
    if v > best_v:
        best_v = v; best_S = set(S)

improved = True
while improved and len(best_S) < k:
    improved = False
    for (v, tr, tc) in tiles[:12]:
        if v <= 0 or (tr, tc) in best_S:
            continue
        dist = {}; prev = {}; pq = []
        for (r, c) in best_S:
            dist[(r, c)] = 0; heapq.heappush(pq, (0, r, c))
        found = False
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist.get((r, c), float("inf")):
                continue
            if (r, c) == (tr, tc):
                found = True; break
            for dr, dc in NB:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    w = 0 if (nr, nc) in best_S else max(0, -net[nr][nc])
                    nd = d + w
                    if nd < dist.get((nr, nc), float("inf")):
                        dist[(nr, nc)] = nd; prev[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (nd, nr, nc))
        if not found:
            continue
        path = []; cur = (tr, tc)
        while cur in prev and cur not in best_S:
            path.append(cur); cur = prev[cur]
        path_new = [p for p in path if p not in best_S]
        if len(best_S) + len(path_new) > k:
            continue
        trial = region_grow(set(best_S) | set(path_new))
        tv = val(trial)
        if tv > best_v:
            best_v = tv; best_S = trial; improved = True
            break

changed = True
while changed and len(best_S) > 1:
    changed = False
    for cell in list(best_S):
        if net[cell[0]][cell[1]] >= 0:
            continue
        T = best_S - {cell}
        if T and connected(T):
            best_S = T; changed = True; break

print(json.dumps({"cells": [[r, c] for (r, c) in best_S]}))
