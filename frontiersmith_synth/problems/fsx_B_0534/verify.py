#!/usr/bin/env python3
# Deterministic checker for firebreak-ignition-lattice (format C, MAXIMIZE
# protected fuel).  CLI: python3 verify.py <in> <out> <ans>   (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
import sys


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


# ---- union-find over non-blocked cells; returns worst ignition component ----
def worst_burned(N, fuel, blocked, ign):
    parent = list(range(N * N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in range(N):
        base = r * N
        for c in range(N):
            i = base + c
            if blocked[i]:
                continue
            if c + 1 < N and not blocked[i + 1]:
                union(i, i + 1)
            if r + 1 < N and not blocked[i + N]:
                union(i, i + N)
    comp = {}
    for r in range(N):
        base = r * N
        for c in range(N):
            i = base + c
            if blocked[i]:
                continue
            root = find(i)
            comp[root] = comp.get(root, 0) + fuel[r][c]
    worst = 0
    for (r, c) in ign:
        i = r * N + c
        if blocked[i]:
            continue  # never happens (enforced), but be safe
        v = comp.get(find(i), 0)
        if v > worst:
            worst = v
    return worst


def main():
    # ---- instance ----------------------------------------------------------
    try:
        it = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad instance")
    p = 0
    N = int(it[p]); F = int(it[p + 1]); K = int(it[p + 2]); p += 3
    ign = []
    ignset = set()
    for _ in range(K):
        r = int(it[p]); c = int(it[p + 1]); p += 2
        ign.append((r, c)); ignset.add(r * N + c)
    fuel = [[0] * N for _ in range(N)]
    T = 0
    for r in range(N):
        for c in range(N):
            v = int(it[p]); p += 1
            fuel[r][c] = v; T += v

    # ---- participant output ------------------------------------------------
    try:
        ot = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not ot:
        fail("empty output")
    try:
        M = int(ot[0])
    except Exception:
        fail("bad M")
    if M < 0 or M > F:
        fail("M out of range (0..F)")
    if len(ot) < 1 + 2 * M:
        fail("truncated firebreak list")

    blocked = [False] * (N * N)
    seen = set()
    for k in range(M):
        try:
            r = int(ot[1 + 2 * k]); c = int(ot[2 + 2 * k])
        except Exception:
            fail("bad firebreak %d" % k)
        if r < 0 or r >= N or c < 0 or c >= N:
            fail("firebreak %d out of grid" % k)
        idx = r * N + c
        if idx in ignset:
            fail("firebreak on an ignition point")
        if idx in seen:
            fail("duplicate firebreak")
        seen.add(idx)
        blocked[idx] = True

    F_obj = T - worst_burned(N, fuel, blocked, ign)

    # ---- internal baseline: one straight wall cutting the map in half ------
    # (a single full vertical firebreak at the free column nearest the middle).
    ign_cols = set(c for (_, c) in ign)
    mid = N // 2
    col = None
    for d in range(N):
        for cand in (mid + d, mid - d):
            if 0 <= cand < N and cand not in ign_cols:
                col = cand; break
        if col is not None:
            break
    bbl = [False] * (N * N)
    if col is not None:
        for r in range(N):
            bbl[r * N + col] = True
    B = T - worst_burned(N, fuel, bbl, ign)
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F_obj / max(1e-9, B))
    print("F_obj=%d B=%.1f T=%d Ratio: %.6f" % (F_obj, B, T, sc / 1000.0))


if __name__ == "__main__":
    main()
