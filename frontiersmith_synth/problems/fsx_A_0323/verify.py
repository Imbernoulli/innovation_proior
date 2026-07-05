#!/usr/bin/env python3
# Deterministic checker for Highway Toll Gantries (format C, maximize sum of radii).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Reads the strip + pylon instance and the participant's disk placement, validates
# feasibility strictly, then scores F = sum(r_i) against an internal single-center-row
# baseline B. Prints "... Ratio: <r>" with r in [0, 1] on its own final line.
import sys, math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    # ---- instance ----
    try:
        it = open(sys.argv[1]).read().split()
        N = int(it[0]); L = float(it[1]); W = float(it[2]); K = int(it[3]); RHO = float(it[4])
        px = []; py = []
        for j in range(K):
            px.append(float(it[5 + 2 * j]))
            py.append(float(it[6 + 2 * j]))
    except Exception:
        fail("bad instance")

    # ---- participant output ----
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
    if M < 0 or M > N:
        fail("M out of range")
    if len(ot) < 1 + 3 * M:
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
        # containment in the strip
        if x - r < -TOL or x + r > L + TOL or y - r < -TOL or y + r > W + TOL:
            fail("disk %d outside strip" % k)
        xs.append(x); ys.append(y); rs.append(r)

    # ---- non-overlap between sensor disks (O(M^2)) ----
    for a in range(M):
        for b in range(a + 1, M):
            dx = xs[a] - xs[b]; dy = ys[a] - ys[b]
            d = math.sqrt(dx * dx + dy * dy)
            if d < rs[a] + rs[b] - TOL:
                fail("disk overlap %d,%d" % (a, b))

    # ---- no sensor disk may overlap a fixed pylon (O(M*K)) ----
    for a in range(M):
        if rs[a] <= 0.0:
            continue
        for j in range(K):
            dx = xs[a] - px[j]; dy = ys[a] - py[j]
            d = math.sqrt(dx * dx + dy * dy)
            if d < rs[a] + RHO - TOL:
                fail("disk %d hits pylon %d" % (a, j))

    F = sum(rs)

    # ---- internal baseline: N equal disks in a single CENTER row ----
    # radius r0 = min(W/4, L/(2N)); the center corridor is guaranteed pylon-free by gen,
    # so this construction is always feasible. B = N * r0.
    r0 = min(W / 4.0, L / (2.0 * N))
    B = N * r0

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
