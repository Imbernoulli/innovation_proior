# TIER: strong
"""The reframe: drawdowns superpose, so the question is never "how do I lower this
one wet cell" but "where does the inflow reaching *all* the wet cells pass through".
We compare three explicit strategies and keep the cheapest:
  (1) one well per foundation cell (the naive frame),
  (2) one well per contiguous wet patch,
  (3) one well per patch-CLUSTER, searched over open cells lying between the
      recharge boundary and the cluster (a graph-distance proxy for "the cut") --
      a single well placed on the shared inflow path can clear several patches at
      once, paying the fixed drilling cost only once instead of once per patch.
Because pumping cost is (fixed_cost + rate^2) per well used, collapsing several
patches onto one well is only worth it when the shared-path well doesn't need a
much higher rate than a local well would -- exactly the well-interference trade-off
the fixed cost is designed to expose."""
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


def required_q(cells_targets, dmap, bd_hub, reach_l, screen_l, qmax):
    need = 0.0
    for cell, t in cells_targets:
        d = dmap.get(cell)
        if d is None:
            return None
        k = kernel(d, bd_hub, reach_l, screen_l)
        if k <= 1e-12:
            return None
        need = max(need, t / k)
    q = math.ceil(need - 1e-9)
    if q < 1:
        q = 1
    if q > qmax:
        return None
    return q


def best_hub(R, C, wall, distR, candidates, cells_targets, reach_l, screen_l, qmax):
    best = None
    for hub in candidates:
        bd = distR.get(hub)
        if bd is None:
            continue
        dmap = bfs_full(R, C, wall, hub)
        q = required_q(cells_targets, dmap, bd, reach_l, screen_l, qmax)
        if q is None:
            continue
        cost = q * q
        if best is None or cost < best[0]:
            best = (cost, hub, q)
    return best


def union_find_blobs(found):
    parent = {v: v for v in found}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for v in found:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            w = (v[0] + dr, v[1] + dc)
            if w in parent:
                union(v, w)

    groups = {}
    for v in found:
        groups.setdefault(find(v), []).append(v)
    return list(groups.values())


def main():
    data = sys.stdin.read().split()
    it = iter(data)
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
        r = int(next(it)); c = int(next(it)); t = float(next(it))
        found.append((r, c)); targets[(r, c)] = t
    reach_l = float(next(it)); screen_l = float(next(it))
    fixed_cost = int(next(it)); qmax = int(next(it)); budget = int(next(it))

    distR = bfs_multi(R, C, wall, recharge)
    distF = bfs_multi(R, C, wall, found)
    open_cells = [(r, c) for r in range(R) for c in range(C)
                  if (r, c) not in recharge and (r, c) not in wall]

    CAND_CAP = 80

    def top_candidates(distF_local):
        scored = []
        for v in open_cells:
            dr = distR.get(v)
            df = distF_local.get(v)
            if dr is None or df is None:
                continue
            scored.append((dr + df, v))
        scored.sort(key=lambda x: x[0])
        return [v for _, v in scored[:CAND_CAP]]

    # (1) per-cell
    percell_wells = []
    percell_cost = 0
    ok1 = True
    for v in found:
        bd = distR.get(v)
        if bd is None:
            ok1 = False
            break
        k = kernel(0, bd, reach_l, screen_l)
        need = targets[v] / max(1e-12, k)
        q = min(qmax, max(1, math.ceil(need - 1e-9)))
        percell_wells.append((v[0], v[1], q))
        percell_cost += fixed_cost + q * q
    if not ok1:
        percell_cost = float("inf")

    # (2) per-blob
    blobs = union_find_blobs(found)
    perblob_wells = []
    perblob_cost = 0
    ok2 = True
    for blob in blobs:
        distF_b = bfs_multi(R, C, wall, blob)
        cand = top_candidates(distF_b)
        cand = list(dict.fromkeys(list(cand) + list(blob)))
        ct = [(v, targets[v]) for v in blob]
        b = best_hub(R, C, wall, distR, cand, ct, reach_l, screen_l, qmax)
        if b is None:
            ok2 = False
            break
        cost, hub, q = b
        perblob_wells.append((hub[0], hub[1], q))
        perblob_cost += fixed_cost + cost
    if not ok2:
        perblob_cost = float("inf")

    # (3) single global hub across the whole cluster
    global_wells = []
    global_cost = float("inf")
    cand_g = top_candidates(distF)
    ct_all = [(v, targets[v]) for v in found]
    bg = best_hub(R, C, wall, distR, cand_g, ct_all, reach_l, screen_l, qmax)
    if bg is not None:
        cost, hub, q = bg
        global_cost = fixed_cost + cost
        global_wells = [(hub[0], hub[1], q)]

    options = [(percell_cost, percell_wells), (perblob_cost, perblob_wells),
               (global_cost, global_wells)]
    options = [o for o in options if o[0] < float("inf") and o[0] <= budget + 1e-9]
    if not options:
        # fall back to whichever is cheapest even if (shouldn't happen) budget-tight
        options = [(percell_cost, percell_wells)]
    options.sort(key=lambda o: o[0])
    _, chosen = options[0]

    out = [str(len(chosen))]
    for (r, c, q) in chosen:
        out.append(f"{r} {c} {q}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
