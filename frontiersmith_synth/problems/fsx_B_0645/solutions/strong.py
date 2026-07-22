# TIER: strong
"""The insight: mixing time is governed by CONDUCTANCE (cut-size vs. the volume on the
smaller side of the cut), not by distance -- and the diameter cap only constrains distance.

If the diameter budget can carry all n rooms as a single straight corridor (n <= D+1), that
corridor IS the extremal slow-mixing shape for a free diameter budget -- use it.

Otherwise the cap forbids a long-enough corridor, so use a HUB-AND-CHAMBER ("multi-throat
dumbbell") layout: a short central hub, with k spokes each = a narrow single-file throat of
length t plus a compact, near-square 2D CHAMBER at its tip. Two chambers reached through the
hub only pay 2*(t + chamber_radius) of diameter budget, and a compact chamber's radius grows
like sqrt(volume) rather than 1:1 with volume like a corridor does -- this decouples "how far"
(diameter) from "how much" (volume). Conductance of the worst cut (severing one chamber) is
~ 1/chamber_volume, so FEWER, LARGER, BALANCED chambers minimize conductance (slowest mixing);
more/smaller chambers are only used when forced by a tighter D. We search a small grid of
(k, t) combinations, keep the ones that are actually BFS-feasible (diameter <= D), score each
with the true objective (same spectral-gap computation the checker uses), and output the best.
"""
import sys
from collections import deque

import numpy as np


def neighbors4(r, c):
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


def build_adj(cells):
    cellset = set(cells)
    return {p: [q for q in neighbors4(*p) if q in cellset] for p in cells}


def bfs_diam(cells):
    adj = build_adj(cells)
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


def relaxation_time(cells):
    cell_list = list(cells)
    idx = {p: i for i, p in enumerate(cell_list)}
    n = len(cell_list)
    adj = build_adj(cell_list)
    deg = np.array([len(adj[p]) for p in cell_list], dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    for p in cell_list:
        for q in adj[p]:
            A[idx[p], idx[q]] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    M = dinv[:, None] * A * dinv[None, :]
    S = 0.5 * np.eye(n) + 0.5 * M
    S = 0.5 * (S + S.T)
    ev = np.linalg.eigvalsh(S)
    ev.sort()
    gap = 1.0 - ev[-2]
    return 1.0 / max(gap, 1e-12)


def compact_rect(n, r0, c0):
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


def straight_path(n, r0, c0):
    return [(r0, c0 + j) for j in range(n)]


def dumbbell(n, hub, t, k, W, H):
    """k axis-aligned spokes from hub; each spoke = t throat cells + a compact chamber."""
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)][:k]
    rem = n - 1 - k * t
    if rem < 0:
        return None
    base = rem // k
    extra = rem - base * k
    vols = [base + (1 if i < extra else 0) for i in range(k)]
    cells = {hub}
    for i, (dr, dc) in enumerate(dirs):
        v = vols[i]
        thr = []
        for s in range(1, t + 1):
            thr.append((hub[0] + dr * s, hub[1] + dc * s))
        cells.update(thr)
        ap = thr[-1] if t > 0 else hub
        pd = (dc, dr)
        if v == 0:
            continue
        best_cells, best_max = None, None
        for a in range(1, v + 1):
            b = -(-v // a)
            cand = []
            for ii in range(1, a + 1):
                for jj in range(-(b // 2), b - (b // 2)):
                    cand.append((ii + abs(jj),
                                 (ap[0] + ii * dr + jj * pd[0], ap[1] + ii * dc + jj * pd[1])))
            cand.sort(key=lambda x: x[0])
            chosen = cand[:v]
            mx = chosen[-1][0] if chosen else 0
            if best_max is None or mx < best_max:
                best_max, best_cells = mx, [c for _, c in chosen]
        cells.update(best_cells)
    if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
        return None
    return list(cells)


def main():
    W, H, n, D = map(int, sys.stdin.read().split())
    hub = (H // 2, W // 2)
    candidates = []

    if n <= D + 1:
        p = straight_path(n, hub[0], max(0, hub[1] - n // 2))
        if all(0 <= c < W for _, c in p):
            d = bfs_diam(p)
            if d is not None and d <= D:
                candidates.append(p)

    for k in (2, 3, 4):
        for t in range(0, D + 1):
            db = dumbbell(n, hub, t, k, W, H)
            if db is None or len(db) != n:
                continue
            d = bfs_diam(db)
            if d is not None and d <= D:
                candidates.append(db)

    # safety fallback: the checker's own baseline is always feasible under D
    fallback = compact_rect(n, max(0, hub[0] - 3), max(0, hub[1] - 3))
    if bfs_diam(fallback) is not None and bfs_diam(fallback) <= D:
        candidates.append(fallback)

    best = max(candidates, key=relaxation_time)

    out = []
    for (r, c) in best:
        out.append(f"{r} {c}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
