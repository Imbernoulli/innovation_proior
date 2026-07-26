# TIER: greedy
"""The obvious "smart" pass: repeatedly find the wettest (most-violated) foundation
cell and pump it, reusing the well already sitting there if one exists. This *does*
benefit a little from superposition between adjacent cells inside the same wet
patch, but it never considers pumping anywhere except a currently-wet foundation
cell -- so with several separated wet patches it still ends up drilling one well
per patch, paying the fixed cost of each, instead of intercepting the shared inflow
upstream."""
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


def field_at(wells, dist_cache, distR, reach_l, screen_l, v):
    tot = 0.0
    for w, q in wells.items():
        dm = dist_cache.get(w)
        if dm is None or v not in dm:
            continue
        tot += q * kernel(dm[v], distR[w], reach_l, screen_l)
    return tot


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

    wells = {}
    dist_cache = {}
    stuck = set()
    max_iters = len(found) * qmax + 20
    for _ in range(max_iters):
        worst = None
        worstv = -1.0
        for v in found:
            if v in stuck:
                continue
            s = field_at(wells, dist_cache, distR, reach_l, screen_l, v)
            d = targets[v] - s
            if d > worstv:
                worstv = d
                worst = v
        if worst is None or worstv <= 1e-9:
            break
        if worst not in dist_cache:
            dist_cache[worst] = bfs_full(R, C, wall, worst)
        cur = wells.get(worst, 0)
        if cur >= qmax:
            stuck.add(worst)
            continue
        wells[worst] = cur + 1

    # respect budget: if over budget, drop wells (highest rate first) until under,
    # (a real coder's quick patch, still naive)
    def energy():
        return sum(fixed_cost + q * q for q in wells.values())

    order = sorted(wells.keys(), key=lambda w: -wells[w])
    idx = 0
    while energy() > budget and idx < len(order):
        del wells[order[idx]]
        idx += 1

    out = [str(len(wells))]
    for (r, c), q in wells.items():
        out.append(f"{r} {c} {q}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
