#!/usr/bin/env python3
# Deterministic checker for "Reservoir Dam Network" (format C, maximize sum of radii).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]. Any feasibility violation -> Ratio: 0.0.
import sys, math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        it = open(sys.argv[1]).read().split()
        N = int(it[0]); W = float(it[1]); H = float(it[2])
        CX = float(it[3]); CY = float(it[4]); Q = float(it[5])
    except Exception:
        fail("bad instance")

    try:
        ot = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if not ot:
        fail("empty output")

    # Guard against pathological huge outputs before doing O(M^2) work.
    if len(ot) > 6 * N + 8:
        fail("too many tokens")

    try:
        M = int(ot[0])
    except Exception:
        fail("bad M")

    if M < 0 or M > N:
        fail("M out of range")

    need = 1 + 3 * M
    if len(ot) < need:
        fail("truncated disks")

    xs = []; ys = []; rs = []
    for k in range(M):
        try:
            x = float(ot[1 + 3 * k]); y = float(ot[2 + 3 * k]); r = float(ot[3 + 3 * k])
        except Exception:
            fail("bad disk %d" % k)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            fail("non-finite disk %d" % k)
        if r < -TOL:
            fail("negative radius %d" % k)
        # containment inside the valley floor
        if x - r < -TOL or x + r > W + TOL or y - r < -TOL or y + r > H + TOL:
            fail("disk %d outside valley" % k)
        # must not overlap the protected wetland
        dcx = x - CX; dcy = y - CY
        dc = math.sqrt(dcx * dcx + dcy * dcy)
        if dc < r + Q - TOL:
            fail("disk %d floods wetland" % k)
        xs.append(x); ys.append(y); rs.append(r)

    # pairwise non-overlap (O(M^2), M small)
    for a in range(M):
        for b in range(a + 1, M):
            dx = xs[a] - xs[b]; dy = ys[a] - ys[b]
            d = math.sqrt(dx * dx + dy * dy)
            if d < rs[a] + rs[b] - TOL:
                fail("overlap %d,%d" % (a, b))

    F = sum(rs)

    # Internal trivial baseline the checker builds itself: a single bottom row of
    # N equal disks that clears the wetland. r_b = min(W/2N, (H/2 - Q)/2); B = N*r_b.
    r_b = min(W / (2.0 * N), (H / 2.0 - Q) / 2.0)
    if r_b <= 0:
        r_b = W / (2.0 * N)
    B = N * r_b

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
