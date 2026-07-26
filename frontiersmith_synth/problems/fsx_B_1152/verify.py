#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for aquifer-well-interference-drawdown.

Reads the instance, reads the participant's well placement, validates strictly,
computes the superposed drawdown field via the same graph-distance kernel used to
build the instance, and scores minimized pump energy against an internal baseline
(the checker's own per-cell-independent-well construction).
"""
import sys
import math
from collections import deque


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


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(tok_iter, n):
    return [int(next(tok_iter)) for _ in range(n)]


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        itoks = f.read().split()
    it = iter(itoks)
    try:
        R = int(next(it)); C = int(next(it))
        n_rech = int(next(it))
        recharge = set()
        for _ in range(n_rech):
            r = int(next(it)); c = int(next(it))
            recharge.add((r, c))
        n_wall = int(next(it))
        wall = set()
        for _ in range(n_wall):
            r = int(next(it)); c = int(next(it))
            wall.add((r, c))
        n_found = int(next(it))
        found = []
        targets = {}
        for _ in range(n_found):
            r = int(next(it)); c = int(next(it))
            t = float(next(it))
            found.append((r, c))
            targets[(r, c)] = t
        reach_l = float(next(it)); screen_l = float(next(it))
        fixed_cost = int(next(it)); qmax = int(next(it)); budget = int(next(it))
    except StopIteration:
        fail("truncated instance (harness bug, not a submission problem)")
        return

    # ---- read participant output strictly ----
    try:
        with open(out_path) as f:
            otext = f.read()
    except Exception:
        fail("cannot read output")
        return
    otoks = otext.split()
    oit = iter(otoks)
    try:
        w_tok = next(oit)
    except StopIteration:
        fail("empty output")
        return
    try:
        W = int(w_tok)
    except ValueError:
        fail("first token not an integer well count")
        return
    if W < 0 or W > 20000:
        fail(f"well count {W} out of range")
        return

    wells = []
    seen_cells = set()
    for i in range(W):
        try:
            rt = next(oit); ct = next(oit); qt = next(oit)
        except StopIteration:
            fail(f"truncated well list at well {i}")
            return
        try:
            r = int(rt); c = int(ct); q = int(qt)
        except ValueError:
            fail(f"well {i} has non-integer fields")
            return
        if not (math.isfinite(r) and math.isfinite(c) and math.isfinite(q)):
            fail(f"well {i} has non-finite field")
            return
        if not (0 <= r < R and 0 <= c < C):
            fail(f"well {i} at ({r},{c}) out of grid")
            return
        if (r, c) in recharge:
            fail(f"well {i} placed on a recharge cell")
            return
        if (r, c) in wall:
            fail(f"well {i} placed on a wall/blocked cell")
            return
        if (r, c) in seen_cells:
            fail(f"duplicate well at ({r},{c})")
            return
        if q < 1 or q > qmax:
            fail(f"well {i} rate {q} out of [1,{qmax}]")
            return
        seen_cells.add((r, c))
        wells.append((r, c, q))

    # trailing extra non-whitespace tokens beyond declared count are tolerated (ignored),
    # matching common convention; but reject if fewer tokens than declared (already caught).

    # ---- energy / budget ----
    E = 0
    for (r, c, q) in wells:
        E += fixed_cost + q * q
    if E > budget + 1e-9:
        fail(f"energy {E} exceeds budget {budget}")
        return

    # ---- drawdown field from submitted wells ----
    distR = bfs_multi(R, C, wall, recharge)
    field = {v: 0.0 for v in found}
    for (r, c, q) in wells:
        w = (r, c)
        bd = distR.get(w)
        if bd is None:
            continue  # well isolated from any recharge cell -> contributes nothing (guarded)
        dmap = bfs_full(R, C, wall, w)
        for v in found:
            d = dmap.get(v)
            if d is None:
                continue
            field[v] += q * kernel(d, bd, reach_l, screen_l)

    for v in found:
        s = field[v]
        if not math.isfinite(s):
            fail(f"non-finite field at {v}")
            return
        if s < targets[v] - 1e-6:
            fail(f"foundation cell {v} drawdown {s:.6f} < target {targets[v]:.6f}")
            return

    # ---- internal baseline B: per-cell independent well, sized in isolation ----
    B = 0
    for v in found:
        bd = distR.get(v)
        if bd is None:
            B += fixed_cost + qmax * qmax
            continue
        k = kernel(0, bd, reach_l, screen_l)
        if k <= 1e-12:
            q = qmax
        else:
            need = targets[v] / k
            q = min(qmax, max(1, math.ceil(need - 1e-9)))
        B += fixed_cost + q * q

    F = E
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print(f"energy={F} baseline={B} wells={W}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
