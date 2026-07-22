#!/usr/bin/env python3
# Deterministic checker for self-trapping-reward-tour (format C, MAXIMIZE reward
# collected by a self-avoiding walk).  CLI: python3 verify.py <in> <out> <ans>
# (ans ignored).  Prints "... Ratio: <r>" with r in [0,1]; any feasibility
# breach -> Ratio: 0.0.
import sys

DIRS = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # canonical baseline priority: R,D,U,L


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_instance(path):
    try:
        it = open(path).read().split()
    except Exception:
        fail("bad instance")
    p = 0
    N = int(it[p]); V = int(it[p + 1]); E = int(it[p + 2]); S = int(it[p + 3]); p += 4
    cells = []
    pos2idx = {}
    for i in range(V):
        r = int(it[p]); c = int(it[p + 1]); rew = int(it[p + 2])
        k = int(it[p + 3]); kid = int(it[p + 4]); p += 5
        cells.append((r, c, rew, k, kid))
        pos2idx[(r, c)] = i
    adj = [set() for _ in range(V)]
    for _ in range(E):
        u = int(it[p]); v = int(it[p + 1]); p += 2
        adj[u].add(v); adj[v].add(u)
    return N, V, cells, adj, S, pos2idx


def baseline_walk(cells, adj, S, pos2idx):
    """Canonical, reward-blind direction-priority walker -> the checker's own
    trivial feasible construction B.  It never looks at reward magnitudes, so
    it naturally prefers the spine (R before D) over decoy spurs, and at the
    fork prefers 'continue right' (the branch WITHOUT the key) -- it starves
    at the locked gate and never reaches the vault."""
    visited = [False] * len(cells)
    visited[S] = True
    keys = set()
    cur = S
    total = cells[S][2]
    while True:
        r, c = cells[cur][0], cells[cur][1]
        moved = False
        for dr, dc in DIRS:
            v = pos2idx.get((r + dr, c + dc))
            if v is None or v not in adj[cur] or visited[v]:
                continue
            _, _, rew, k, kid = cells[v]
            if k == 2 and kid not in keys:
                continue
            visited[v] = True
            if k == 1:
                keys.add(kid)
            total += rew
            cur = v
            moved = True
            break
        if not moved:
            return total


def main():
    N, V, cells, adj, S, pos2idx = parse_instance(sys.argv[1])
    T = sum(c[2] for c in cells)

    # ---- participant output ----
    try:
        ot = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not ot:
        fail("empty output")
    try:
        L = int(ot[0])
    except Exception:
        fail("bad L")
    if L < 1 or L > V:
        fail("L out of range [1, V]")
    if len(ot) < 1 + 2 * L:
        fail("truncated path")

    path = []
    for i in range(L):
        try:
            r = int(ot[1 + 2 * i]); c = int(ot[2 + 2 * i])
        except Exception:
            fail("bad coordinate at step %d" % i)
        if r < 0 or r >= N or c < 0 or c >= N:
            fail("coordinate out of grid at step %d" % i)
        if (r, c) not in pos2idx:
            fail("cell (%d,%d) is not a declared world cell (step %d)" % (r, c, i))
        path.append(pos2idx[(r, c)])

    if path[0] != S:
        fail("path must start at the given start cell")

    seen = set()
    keys = set()
    F = 0
    for i, u in enumerate(path):
        if u in seen:
            fail("revisits cell index %d (self-avoiding violated) at step %d" % (u, i))
        if i > 0:
            prev = path[i - 1]
            if u not in adj[prev]:
                fail("step %d -> %d is not a declared corridor edge" % (prev, u))
        r, c, rew, k, kid = cells[u]
        if k == 2 and kid not in keys:
            fail("entered locked gate (needs key %d) at step %d" % (kid, i))
        seen.add(u)
        if k == 1:
            keys.add(kid)
        F += rew

    B = baseline_walk(cells, adj, S, pos2idx)
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d T=%d Ratio: %.6f" % (F, B, T, sc / 1000.0))


if __name__ == "__main__":
    main()
