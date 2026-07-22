# TIER: strong
"""Insight: above the percolation threshold the flammable footprint is one
giant connected cluster, so expected burned area is governed by the size of
the piece a fire's starting cluster gets fragmented into -- NOT by which
single corridor out of the historical hot zone is blocked. This exploits a
graph-separator fact: in an UNWEIGHTED graph, BFS distance changes by at most
1 across any edge, so removing every cell at exact BFS-distance d from some
source cell fully disconnects {dist < d} from {dist > d} inside that
component (no edge can skip a whole distance level). So: find the giant
component(s), and repeatedly bisect the currently-largest fragment by
picking the BFS-distance layer (from an extreme corner of that fragment)
that (a) fits the remaining budget and (b) best balances the two resulting
sides -- i.e. minimizes the worse side's size, which is exactly what caps
the expected damage from an ignition landing anywhere. Recurse until the
budget is spent or no fragment is worth cutting further. This is a genuine
separator/decomposition argument, not a tuned version of the greedy corridor
wall -- it never even looks at the hot zone or the wind.
"""
import sys, json
from collections import deque
import heapq

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIN_COMP_TO_CUT = 6


def _bfs_dist(grid, R, C, removed, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        d0 = dist[(r, c)]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1 \
               and (nr, nc) not in removed and (nr, nc) not in dist:
                dist[(nr, nc)] = d0 + 1
                q.append((nr, nc))
    return dist


def _component_of(grid, R, C, removed, start):
    comp = [start]
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1 \
               and (nr, nc) not in removed and (nr, nc) not in seen:
                seen.add((nr, nc))
                comp.append((nr, nc))
                q.append((nr, nc))
    return comp


def _all_components(grid, R, C, removed):
    seen = set()
    comps = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 1 and (r, c) not in removed and (r, c) not in seen:
                comp = _component_of(grid, R, C, removed, (r, c))
                seen.update(comp)
                comps.append(comp)
    return comps


def _best_layer_cut(grid, R, C, removed, comp, remaining_budget):
    """Best BFS-distance layer to remove from `comp`, fitting the budget,
    minimizing the larger of the two resulting side sizes. None if nothing
    useful/affordable is found."""
    if len(comp) < MIN_COMP_TO_CUT or remaining_budget <= 0:
        return None
    src = min(comp, key=lambda x: (x[0] + x[1], x[0]))
    dist = _bfs_dist(grid, R, C, removed, src)
    if not dist:
        return None
    maxd = max(dist.values())
    layers = {}
    for cell, d in dist.items():
        layers.setdefault(d, []).append(cell)
    n = len(dist)
    best = None
    for d in range(1, maxd):
        layer = layers.get(d, [])
        if not layer or len(layer) > remaining_budget:
            continue
        inside = sum(1 for dd in dist.values() if dd < d)
        outside = n - inside - len(layer)
        if inside <= 0 or outside <= 0:
            continue
        worst = max(inside, outside)
        key = (worst, len(layer))
        if best is None or key < best[0]:
            best = (key, layer)
    return best[1] if best else None


def solve(inst):
    R, C = inst["R"], inst["C"]
    grid = inst["flammable"]
    budget = inst["budget"]
    removed = set()
    remaining = budget

    comps = _all_components(grid, R, C, removed)
    heap = [(-len(comp), comp) for comp in comps if len(comp) >= MIN_COMP_TO_CUT]
    heapq.heapify(heap)

    while heap and remaining > 0:
        negsize, comp = heapq.heappop(heap)
        comp = [cell for cell in comp if cell not in removed]
        if len(comp) < MIN_COMP_TO_CUT:
            continue
        cut = _best_layer_cut(grid, R, C, removed, comp, remaining)
        if not cut:
            continue
        for cell in cut:
            if cell not in removed and remaining > 0:
                removed.add(cell)
                remaining -= 1
        for sub in _all_components(grid, R, C, removed):
            s = set(sub)
            if s <= set(comp) and len(sub) >= MIN_COMP_TO_CUT:
                heapq.heappush(heap, (-len(sub), sub))

    cells = [[r, c] for (r, c) in removed]
    if len(cells) > budget:
        cells = cells[:budget]
    return cells


inst = json.load(sys.stdin)
print(json.dumps({"cells": solve(inst)}))
