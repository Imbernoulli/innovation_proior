#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   deterministic scorer (ans ignored).

Reads the instance (N, r0) from <in> and the participant artifact from <out>:
N lines "x y r" describing dome circles. Validates feasibility STRICTLY with a
fixed 1e-6 tolerance, then scores objective F = sum of radii, normalized against an
internal trivial grid baseline B. Prints exactly one 'Ratio: <v>' line.
"""
import sys
import math

EPS = 1e-6
CX, CY = 0.5, 0.5


def read_instance(path):
    toks = open(path).read().split()
    N = int(toks[0])
    r0 = float(toks[1])
    return N, r0


def baseline(N, r0):
    """A trivial feasible construction: small equal circles on a coarse grid,
    skipping cells that would touch the antenna keep-out disk. Deterministic."""
    G = int(math.ceil(math.sqrt(N))) + 2
    rb = 0.3 / G
    out = []
    for i in range(G):
        for j in range(G):
            if len(out) >= N:
                break
            x = (i + 0.5) / G
            y = (j + 0.5) / G
            if math.hypot(x - CX, y - CY) >= r0 + rb + 1e-12:
                out.append((x, y, rb))
        if len(out) >= N:
            break
    return out


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    N, r0 = read_instance(inf)

    try:
        vals = list(map(float, open(outf).read().split()))
    except Exception:
        fail("parse error")

    if len(vals) != 3 * N:
        fail("expected %d numbers, got %d" % (3 * N, len(vals)))

    circ = [(vals[3 * k], vals[3 * k + 1], vals[3 * k + 2]) for k in range(N)]

    for (x, y, r) in circ:
        if not (r > 0.0) or math.isnan(r) or math.isinf(r):
            fail("nonpositive/invalid radius")
        if x - r < -EPS or x + r > 1.0 + EPS or y - r < -EPS or y + r > 1.0 + EPS:
            fail("dome leaves the ice pad")
        if math.hypot(x - CX, y - CY) < r0 + r - EPS:
            fail("dome overlaps antenna keep-out")

    for a in range(N):
        xa, ya, ra = circ[a]
        for b in range(a + 1, N):
            xb, yb, rb = circ[b]
            if math.hypot(xa - xb, ya - yb) < ra + rb - EPS:
                fail("two domes overlap")

    F = sum(r for _, _, r in circ)
    B = sum(r for _, _, r in baseline(N, r0))

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
