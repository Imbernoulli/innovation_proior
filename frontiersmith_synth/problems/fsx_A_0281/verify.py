#!/usr/bin/env python3
"""
Deterministic checker for the "Aurora relay grid" (corner-free set) problem.

Instance: toroidal grid Z_n x Z_n with a set of blocked cells.
Artifact (participant output): a set of chosen cells.

Feasibility (all enforced strictly; ANY violation -> Ratio: 0.0):
  * output parses as an integer count s followed by exactly 2*s integer tokens,
  * 0 <= s <= n*n, every coordinate an integer in [0, n),
  * no non-finite tokens (int parse rejects nan/inf),
  * no duplicate cells, no chosen cell is blocked,
  * the set is CORNER-FREE on the torus: there is NO d in {1..n-1} and cell (x,y)
    with (x,y), ((x+d)%n, y) and (x,(y+d)%n) all chosen.

Objective (maximize): F = |chosen set|.
Baseline B (built by the checker): the largest single row's unblocked-cell count
(a single row is always corner-free).  Score:
    sc = min(1000, 100 * F / max(1e-9, B)); print  Ratio: sc/1000
=> a full single row scores ~0.1; you must beat it multiplicatively.
"""
import sys


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(path):
    with open(path) as f:
        data = f.read()
    toks = data.split()
    return toks


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        sys.exit(0)
    inf, outf = sys.argv[1], sys.argv[2]

    # ---- instance ----
    itoks = read_ints(inf)
    idx = 0
    n = int(itoks[idx]); idx += 1
    k = int(itoks[idx]); idx += 1
    blocked = set()
    for _ in range(k):
        x = int(itoks[idx]); y = int(itoks[idx + 1]); idx += 2
        blocked.add((x, y))

    # ---- baseline B = biggest unblocked single row ----
    B = 0
    for y in range(n):
        c = sum(1 for x in range(n) if (x, y) not in blocked)
        if c > B:
            B = c
    if B <= 0:
        B = 1

    # ---- participant artifact ----
    toks = read_ints(outf)
    if len(toks) == 0:
        fail("empty output")
    # first token = declared count
    try:
        s = int(toks[0])
    except (ValueError, TypeError):
        fail("count not an integer (nan/inf/garbage?)")
    if s < 0 or s > n * n:
        fail("count out of range")
    if len(toks) != 1 + 2 * s:
        fail("token count mismatch: expected %d got %d" % (1 + 2 * s, len(toks)))

    chosen = []
    seen = set()
    j = 1
    for _ in range(s):
        ta, tb = toks[j], toks[j + 1]; j += 2
        try:
            x = int(ta); y = int(tb)
        except (ValueError, TypeError):
            fail("coordinate not an integer (nan/inf/garbage?)")
        if x < 0 or x >= n or y < 0 or y >= n:
            fail("coordinate out of grid")
        if (x, y) in blocked:
            fail("chosen a blocked cell (%d,%d)" % (x, y))
        if (x, y) in seen:
            fail("duplicate cell (%d,%d)" % (x, y))
        seen.add((x, y))
        chosen.append((x, y))

    # ---- corner-free check (torus) ----
    S = seen
    for (x, y) in chosen:
        for d in range(1, n):
            if ((x + d) % n, y) in S and (x, (y + d) % n) in S:
                fail("corner at (%d,%d) with d=%d" % (x, y, d))

    F = len(chosen)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d n=%d" % (F, B, n))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
