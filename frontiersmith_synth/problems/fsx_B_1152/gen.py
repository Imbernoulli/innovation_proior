#!/usr/bin/env python3
"""gen.py <testId> -- emits one aquifer-well-interference instance to stdout.

Deterministic in testId only (seeded RNG).  Geometry: a recharge boundary (row 0,
"the old river") feeds water south through a single narrow gap in a rock wall, into
an open channel row, which fans out into several fully separated corridors, each
ending at one foundation "blob" (a patch of the crypt floor).  Blobs are graph-far
from each other (must detour back through the gap) but all graph-close to the
channel cells right below the gap -- the interception line.
"""
import sys
import math
import random
from collections import deque

REACH_L = 8.0
SCREEN_L = 6.0
FIXED_COST = 20
TARGET_MARGIN_BASE = 0.58


def bfs_full(R, C, wall, src):
    dist = {src: 0}
    dq = deque([src])
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in wall and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                dq.append((nr, nc))
    return dist


def bfs_multi(R, C, wall, sources):
    dist = {}
    dq = deque()
    for s in sources:
        dist[s] = 0
        dq.append(s)
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in wall and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                dq.append((nr, nc))
    return dist


def kernel(dist, boundary_dist_w, reach_l, screen_l):
    screening = boundary_dist_w / (boundary_dist_w + screen_l)
    return screening * reach_l / (reach_l + dist)


def build(R, C, wall_row, gap_cols, corridor_row0, blob_row, bands, blob_w):
    recharge = {(0, c) for c in range(C)}
    wall = set()
    for c in range(C):
        if c not in gap_cols:
            wall.add((wall_row, c))
    band_cols = set()
    for lo, hi in bands:
        for c in range(lo, hi + 1):
            band_cols.add(c)
    for r in range(corridor_row0, blob_row + 1):
        for c in range(C):
            if c not in band_cols:
                wall.add((r, c))
    found = []
    for lo, hi in bands:
        mid = (lo + hi) // 2
        bc0 = mid - blob_w // 2
        bc0 = max(lo, min(bc0, hi - blob_w + 1))
        for dc in range(blob_w):
            found.append((blob_row, bc0 + dc))
    return recharge, wall, found


def instance_params(test_id):
    nblob_list = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
    nblob = nblob_list[test_id - 1]
    band_w = 3
    blob_w = 2
    gap_width = 2 if test_id in (2, 6) else 1
    R = 13 if test_id <= 8 else 15
    wall_row = 4
    corridor_row0 = 6
    blob_row = R - 1
    qmax = 10 + (test_id % 4)
    margin = TARGET_MARGIN_BASE + 0.015 * (test_id % 5)
    return dict(nblob=nblob, band_w=band_w, blob_w=blob_w, gap_width=gap_width,
                R=R, wall_row=wall_row, corridor_row0=corridor_row0, blob_row=blob_row,
                qmax=qmax, margin=margin)


def make_instance(test_id):
    rng = random.Random(1000 + test_id)
    p = instance_params(test_id)
    R = p["R"]
    wall_row = p["wall_row"]
    corridor_row0 = p["corridor_row0"]
    blob_row = p["blob_row"]
    band_w = p["band_w"]
    blob_w = p["blob_w"]
    nblob = p["nblob"]
    qmax = p["qmax"]
    margin = p["margin"]

    bands = []
    c = 1
    band_gap = 2 + (rng.randint(0, 1))
    for _ in range(nblob):
        bands.append((c, c + band_w - 1))
        c += band_w + band_gap
    C = c + 2

    gap_center = C // 2
    gw = p["gap_width"]
    gap_cols = set(range(gap_center - gw // 2, gap_center - gw // 2 + gw))

    recharge, wall, found = build(R, C, wall_row, gap_cols, corridor_row0, blob_row, bands, blob_w)

    distR = bfs_multi(R, C, wall, recharge)
    gap_cell = (wall_row, min(gap_cols))
    dmap_gap = bfs_full(R, C, wall, gap_cell)

    targets = {}
    for v in found:
        d = dmap_gap.get(v, 10 ** 6)
        k = kernel(d, distR[gap_cell], REACH_L, SCREEN_L)
        t = max(0.05, qmax * k * margin)
        targets[v] = round(t, 6)

    # internal baseline (per-cell independent well) to size the energy budget
    B = 0
    for v in found:
        dm = bfs_full(R, C, wall, v)
        k = kernel(0, distR[v], REACH_L, SCREEN_L)
        need = targets[v] / max(1e-12, k)
        q = min(qmax, max(1, math.ceil(need - 1e-9)))
        B += FIXED_COST + q * q

    fixed_cost = FIXED_COST
    reach_l = REACH_L
    screen_l = SCREEN_L
    budget = B

    return R, C, sorted(recharge), sorted(wall), sorted(found), targets, reach_l, screen_l, fixed_cost, qmax, budget


def main():
    test_id = int(sys.argv[1])
    R, C, recharge, wall, found, targets, reach_l, screen_l, fixed_cost, qmax, budget = make_instance(test_id)

    out = []
    out.append(f"{R} {C}")
    out.append(str(len(recharge)))
    for (r, c) in recharge:
        out.append(f"{r} {c}")
    out.append(str(len(wall)))
    for (r, c) in wall:
        out.append(f"{r} {c}")
    out.append(str(len(found)))
    for (r, c) in found:
        out.append(f"{r} {c} {targets[(r,c)]:.6f}")
    out.append(f"{reach_l:.6f} {screen_l:.6f} {fixed_cost} {qmax} {budget}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
