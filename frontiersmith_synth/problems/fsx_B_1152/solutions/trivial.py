# TIER: trivial
"""Naive baseline: treat each wet foundation cell in total isolation. For every
foundation cell, drop a single well exactly on top of it, sized (assuming it were
the *only* well in the aquifer) to just clear that cell's own target. Ignores that
neighbouring wells already help each other (superposition) and that many separate
wells each pay the fixed drilling/energy overhead."""
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

    wells = []
    for v in found:
        bd = distR.get(v, 10 ** 6)
        k = kernel(0, bd, reach_l, screen_l)
        need = targets[v] / max(1e-12, k)
        q = min(qmax, max(1, math.ceil(need - 1e-9)))
        wells.append((v[0], v[1], q))

    out = [str(len(wells))]
    for (r, c, q) in wells:
        out.append(f"{r} {c} {q}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
