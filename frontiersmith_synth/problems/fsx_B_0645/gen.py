#!/usr/bin/env python3
"""gen.py <testId> -- prints one 'slow-mix-atrium' museum-floorplan instance to stdout.

Instance = "W H n D": a W x H grid, a budget of n cells to open as exhibit rooms, and a
hard diameter cap D on the induced 4-adjacency graph of the open cells (graph-hop distance,
not Euclidean). Deterministic: fully keyed off testId, no external randomness.

D is derived as (diameter of the checker's own compact near-square baseline for n) + slack,
so the baseline is ALWAYS feasible under D; the slack is what makes a case "tight" (trap,
slack=0..1: only hub/multi-chamber layouts fit) or "loose" (slack large: a corridor/path or
a generous 2-chamber dumbbell fits).
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


def compact_rect(n, r0=1, c0=1):
    """Row-major fill of the near-square rectangle (minimizing rows+cols) holding n cells."""
    best = None
    for rows in range(1, n + 1):
        cols = -(-n // rows)
        if best is None or rows + cols < best[0] + best[1]:
            best = (rows, cols)
    rows, cols = best
    cells = []
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            if cnt >= n:
                break
            cells.append((r0 + i, c0 + j))
            cnt += 1
        if cnt >= n:
            break
    return cells


# (n, slack) per testId 1..10: small->large, mixing loose (slack>0) and tight-diameter
# trap cases (slack=0/1) where a naive corridor cannot carry all n cells.
PLAN = {
    1: (6, 3),
    2: (9, 2),
    3: (14, 3),
    4: (18, 0),
    5: (22, 1),
    6: (26, 3),
    7: (30, 0),
    8: (34, 2),
    9: (40, 2),
    10: (48, 2),
}


def main():
    tid = int(sys.argv[1])
    n, slack = PLAN[tid]
    rect = compact_rect(n)
    base_diam = bfs_diam(rect)
    D = base_diam + slack
    import math
    side = math.isqrt(n)
    while side * side < n:
        side += 1
    margin = side + 10
    W = H = 2 * margin + 5
    print(f"{W} {H} {n} {D}")


if __name__ == "__main__":
    main()
