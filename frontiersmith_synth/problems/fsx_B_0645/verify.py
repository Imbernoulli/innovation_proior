#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the slow-mix-atrium museum
floorplan problem.

Feasibility (strict, prints `Ratio: 0.0` + reason on ANY violation):
  - output has exactly 2*n integer tokens (n coordinate pairs "r c"); non-finite/non-integer
    tokens, wrong count, out-of-range coordinates, or duplicate cells -> reject.
  - the induced 4-adjacency graph of the n open cells must be CONNECTED.
  - its graph diameter (max shortest-path hop distance over all pairs) must be <= D.

Objective F = relaxation time 1/gap of the LAZY random walk on the induced graph: from an
open cell, stay with probability 1/2, else step to a uniformly random OPEN 4-neighbor
(probability 1/(2*deg)). This chain is reversible w.r.t. pi(v) ~ deg(v); its transition
operator is similar to the symmetric matrix S = 0.5*I + 0.5*D^-1/2 A D^-1/2 (same spectrum).
gap = 1 - (second-largest eigenvalue of S), computed by exact (deterministic) dense
eigen-decomposition -- this is the "spectral-gap-minimization" mechanism. F = 1/gap.

Baseline B: the checker's own compact near-square rectangle of n cells (same construction
gen.py used to size D), scored the same way. Score:
  sc = min(1000, 100*F/B);  print("... Ratio: %.6f" % (sc/1000.0))
so reproducing the baseline scores ~0.1 and a 10x-better mixing time saturates at 1.0.
"""
import sys
import math
from collections import deque

import numpy as np


def neighbors4(r, c):
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


def build_adj(cells):
    cellset = set(cells)
    return {p: [q for q in neighbors4(*p) if q in cellset] for p in cells}


def bfs_diam_and_connected(cells):
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
            return None, False
        m = max(dist.values())
        if m > maxd:
            maxd = m
    return maxd, True


def relaxation_time(cells):
    """1 / spectral gap of the lazy random walk on the induced graph. cells must already be
    verified connected. Deterministic dense symmetric eigendecomposition (LAPACK syevd)."""
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
    S = 0.5 * (S + S.T)  # enforce exact symmetry against fp asymmetry
    ev = np.linalg.eigvalsh(S)
    ev.sort()
    lam2 = ev[-2]
    gap = 1.0 - lam2
    return 1.0 / max(gap, 1e-12)


def compact_rect(n, r0=1, c0=1):
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


def reject(msg):
    print("INFEASIBLE: %s Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        W, H, n, D = map(int, f.read().split())

    try:
        with open(outf) as f:
            raw = f.read(10_000_000)
    except Exception:
        reject("cannot read output")
        return

    toks = raw.split()
    if len(toks) > 2 * n + 8:
        reject("too many tokens")
        return
    if len(toks) != 2 * n:
        reject("expected %d tokens (n coordinate pairs), got %d" % (2 * n, len(toks)))
        return

    cells = []
    for i in range(n):
        rt, ct = toks[2 * i], toks[2 * i + 1]
        try:
            r = int(rt)
            c = int(ct)
        except ValueError:
            reject("non-integer token")
            return
        if not (0 <= r < H and 0 <= c < W):
            reject("cell (%d,%d) out of bounds" % (r, c))
            return
        cells.append((r, c))

    if len(set(cells)) != n:
        reject("duplicate cells")
        return

    diam, connected = bfs_diam_and_connected(cells)
    if not connected:
        reject("induced graph is disconnected")
        return
    if diam > D:
        reject("diameter %d exceeds cap D=%d" % (diam, D))
        return

    F = relaxation_time(cells)
    if not math.isfinite(F) or F <= 0:
        reject("non-finite objective")
        return

    base = compact_rect(n)
    B = relaxation_time(base)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("OK n=%d diam=%d F=%.6f B=%.6f Ratio: %.6f" % (n, diam, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
