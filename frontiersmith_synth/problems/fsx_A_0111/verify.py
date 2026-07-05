#!/usr/bin/env python3
"""Deterministic checker for SkyGrid Swarm (corner-free set).

CLI: python3 verify.py <in> <out> <ans>   (ans is ignored placeholder)

Feasibility (any violation -> Ratio: 0.0):
  - each pad in-grid integer coords
  - pairwise distinct
  - none obstructed
  - no corner {(r,c),(r+d,c),(r,c+d)} d>=1 all active

Score (maximization): B = largest single unobstructed row/column line.
  sc = min(1000, 100*F/max(1e-9,B)); Ratio = sc/1000.
"""
import sys


def fail(msg):
    print("Invalid: %s Ratio: 0.0" % msg)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    m = int(next(it))
    b = int(next(it))
    blocked = set()
    for _ in range(b):
        r = int(next(it))
        c = int(next(it))
        blocked.add((r, c))
    return m, blocked


def baseline(m, blocked):
    # largest single line (row or col) of unobstructed pads
    best = 0
    for r in range(m):
        cnt = sum(1 for c in range(m) if (r, c) not in blocked)
        best = max(best, cnt)
    for c in range(m):
        cnt = sum(1 for r in range(m) if (r, c) not in blocked)
        best = max(best, cnt)
    return best


def has_corner(P):
    # group active pads by column; for each pair in a column at rows r1<r2 (d=r2-r1)
    # check whether pivot (r1, c+d) is also active.
    cols = {}
    for (r, c) in P:
        cols.setdefault(c, []).append(r)
    for c, rows in cols.items():
        rs = sorted(rows)
        n = len(rs)
        for i in range(n):
            ri = rs[i]
            for j in range(i + 1, n):
                d = rs[j] - ri
                if (ri, c + d) in P:
                    return True
    return False


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> [ans]  Ratio: 0.0")
        sys.exit(0)
    m, blocked = read_instance(sys.argv[1])

    try:
        with open(sys.argv[2]) as f:
            toks = f.read().split()
    except Exception:
        fail("could not read output.")

    if not toks:
        fail("empty output.")

    it = iter(toks)
    try:
        k = int(next(it))
    except Exception:
        fail("first token must be integer k.")
    if k < 0:
        fail("k negative.")

    pads = []
    try:
        for _ in range(k):
            r = int(next(it))
            c = int(next(it))
            pads.append((r, c))
    except StopIteration:
        fail("fewer than k pads provided.")
    except Exception:
        fail("non-integer coordinate.")

    P = set()
    for (r, c) in pads:
        if not (0 <= r < m and 0 <= c < m):
            fail("pad (%d,%d) out of grid." % (r, c))
        if (r, c) in blocked:
            fail("pad (%d,%d) is obstructed." % (r, c))
        if (r, c) in P:
            fail("duplicate pad (%d,%d)." % (r, c))
        P.add((r, c))

    if has_corner(P):
        fail("output contains a corner.")

    F = len(P)
    B = baseline(m, blocked)
    if B <= 0:
        B = 1
    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
