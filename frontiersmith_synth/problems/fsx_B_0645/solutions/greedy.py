# TIER: greedy
"""The 'obvious' first idea: mixing time feels like it should track how FAR APART the rooms
are, so (1) build the longest straight corridor the diameter cap allows, then (2) once the
budget of n rooms is not used up by that corridor, just keep adding rooms wherever they don't
push the diameter over the cap -- preferring whichever addition keeps the layout as SPREAD OUT
(large diameter) as possible, i.e. still chasing distance, never reasoning about conductance
or chamber balance. This is a real local search (not a closed-form recipe) but it never
discovers the balanced-throat idea, so on cases where the volume badly exceeds what a single
corridor can carry within D, its padding ends up unbalanced and mixes much faster than the
reference dumbbell/hub layout.
"""
import sys
from collections import deque


def neighbors4(r, c):
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


def bfs_diam(cells):
    cellset = set(cells)
    adj = {p: [q for q in neighbors4(*p) if q in cellset] for p in cells}
    n = len(cells)
    maxd = 0
    for s in cells:
        dist = {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        if len(dist) != n:
            return None
        m = max(dist.values())
        if m > maxd:
            maxd = m
    return maxd


def main():
    W, H, n, D = map(int, sys.stdin.read().split())
    start = (H // 2, W // 2)
    cells = {start}
    Lmax = min(n, D + 1)
    cur = start
    length = 1
    while length < Lmax:
        nxt = (cur[0], cur[1] + 1)
        if not (0 <= nxt[1] < W):
            break
        cells.add(nxt)
        cur = nxt
        length += 1

    def frontier():
        f = set()
        for p in cells:
            for q in neighbors4(*p):
                if q not in cells and 0 <= q[0] < H and 0 <= q[1] < W:
                    f.add(q)
        return f

    remaining = n - len(cells)
    for _ in range(remaining):
        cand = sorted(frontier())
        if not cand:
            break
        best_cell, best_diam, best_feas = None, None, None
        for c in cand:
            d = bfs_diam(cells | {c})
            feas = d is not None and d <= D
            if feas:
                if best_feas is not True or d > best_diam:
                    best_cell, best_diam, best_feas = c, d, True
            elif best_feas is not True:
                if best_diam is None or (d is not None and d < best_diam):
                    best_cell, best_diam, best_feas = c, d, False
        if best_cell is None:
            break
        cells.add(best_cell)

    out = []
    for (r, c) in cells:
        out.append(f"{r} {c}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
