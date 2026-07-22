#!/usr/bin/env python3
"""Checker for fsx_A_0947 -- Rising Water, Staged Barriers.
Usage: chk.py <in> <out> <ans>   (ans unused; scorer problem).
Always exits 0; prints "... Ratio: <float>" -- the harness parses the LAST such token.
"""
import sys
import math
from collections import deque

EPS = 1.0
MAX_WALLS_CAP = None  # set after reading R,C


def fail(reason, F=0.0, B=1.0):
    print(f"WA F={F:.6f} B={B:.6f} reason={reason} Ratio: 0.000000")
    sys.exit(0)


def read_tokens(path):
    try:
        with open(path, "r") as f:
            return f.read().split()
    except Exception:
        return None


def simulate(R, C, elev, K, levels, walls_by_stage):
    """Returns final `wet` R x C boolean grid after all K stages, given a per-stage
    list of newly-built wall edges (each a frozenset of two (r,c) tuples)."""
    wet = [[False] * C for _ in range(R)]
    built = set()
    dq = deque()
    for r in range(R):
        for c in range(C):
            if elev[r][c] <= 0:
                wet[r][c] = True
                dq.append((r, c))
    for k in range(K):
        built |= walls_by_stage[k]
        level = levels[k]
        # seed queue with every currently-wet cell (cheap: R*C bounded, K small)
        dq = deque((r, c) for r in range(R) for c in range(C) if wet[r][c])
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and not wet[nr][nc] and elev[nr][nc] <= level:
                    e = frozenset(((r, c), (nr, nc)))
                    if e in built:
                        continue
                    wet[nr][nc] = True
                    dq.append((nr, nc))
    return wet


def main():
    inf, ouf_path = sys.argv[1], sys.argv[2]
    itok = read_tokens(inf)
    it = iter(itok)
    R = int(next(it)); C = int(next(it)); K = int(next(it))
    elev = [[int(next(it)) for _ in range(C)] for _ in range(R)]
    NB = int(next(it))
    p_groups = []
    for _ in range(NB):
        sz = int(next(it))
        cells = [(int(next(it)), int(next(it))) for _ in range(sz)]
        p_groups.append(cells)
    Q = int(next(it))
    f_cells = [(int(next(it)), int(next(it))) for _ in range(Q)]
    levels = [int(next(it)) for _ in range(K)]
    W = [int(next(it)) for _ in range(K)]
    alpha = float(next(it))

    CumW = []
    s = 0
    for k in range(K):
        s += W[k]
        CumW.append(s)

    # ---- baseline B: internal trivial construction (build no walls at all) ----
    empty_by_stage = [set() for _ in range(K)]
    wet_b = simulate(R, C, elev, K, levels, empty_by_stage)

    def score_of(wet):
        p_terms = []
        for grp in p_groups:
            dry = sum(1 for (r, c) in grp if not wet[r][c])
            p_terms.append(dry / len(grp))
        f_dry = sum(1 for (r, c) in f_cells if not wet[r][c])
        f_term = f_dry / len(f_cells) if f_cells else 0.0
        return EPS + sum(p_terms) + alpha * f_term

    B = score_of(wet_b)
    if B <= 0 or not math.isfinite(B):
        B = 1e-6

    otok = read_tokens(ouf_path)
    if otok is None:
        fail("cannot read output", 0.0, B)
    oit = iter(otok)

    def next_int(lo, hi, name):
        try:
            tok = next(oit)
        except StopIteration:
            fail(f"missing token for {name}", 0.0, B)
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token for {name}: {tok!r}", 0.0, B)
        if not (lo <= v <= hi):
            fail(f"{name}={v} out of range [{lo},{hi}]", 0.0, B)
        return v

    max_walls = 4 * R * C + 10
    M = next_int(0, max_walls, "M")

    seen_edges = set()
    walls_by_stage = [set() for _ in range(K)]
    cnt_by_stage = [0] * K
    for i in range(M):
        st = next_int(1, K, f"wall[{i}].stage")
        r1 = next_int(0, R - 1, f"wall[{i}].r1")
        c1 = next_int(0, C - 1, f"wall[{i}].c1")
        r2 = next_int(0, R - 1, f"wall[{i}].r2")
        c2 = next_int(0, C - 1, f"wall[{i}].c2")
        if (r1, c1) == (r2, c2):
            fail(f"wall[{i}] self-loop at ({r1},{c1})", 0.0, B)
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            fail(f"wall[{i}] endpoints not 4-adjacent: ({r1},{c1})-({r2},{c2})", 0.0, B)
        key = frozenset(((r1, c1), (r2, c2)))
        if key in seen_edges:
            fail(f"wall[{i}] duplicate edge ({r1},{c1})-({r2},{c2})", 0.0, B)
        seen_edges.add(key)
        walls_by_stage[st - 1].add(key)
        cnt_by_stage[st - 1] += 1

    # trailing garbage check
    try:
        extra = next(oit)
        fail(f"trailing token {extra!r}", 0.0, B)
    except StopIteration:
        pass

    # cumulative per-stage budget check
    running = 0
    for k in range(K):
        running += cnt_by_stage[k]
        if running > CumW[k]:
            fail(f"cumulative walls {running} > budget {CumW[k]} by stage {k + 1}", 0.0, B)

    wet_f = simulate(R, C, elev, K, levels, walls_by_stage)
    F = score_of(wet_f)
    if not math.isfinite(F):
        fail("non-finite score", 0.0, B)

    sc = min(1000.0, 100.0 * F / max(B, 1e-9))
    ratio = sc / 1000.0
    print(f"OK F={F:.6f} B={B:.6f} M={M} Ratio: {ratio:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
