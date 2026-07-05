# verify.py -- deterministic scorer for the drone-delivery-swarm safety-bubble packing problem.
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints exactly one "Ratio: <float in [0,1]>" line and exits 0.
#
# Feasibility (tol = 1e-6): 0<=M<=N; each r>=-tol and finite; each bubble lies inside the
# circular airspace (dist(center,C)+r <= R); no bubble overlaps any no-fly zone; bubbles are
# pairwise non-overlapping. Objective F = sum of radii (maximize). Ratio = min(1, 0.1*F/B)
# where B is the checker's own airspace-aware equal-radius grid baseline.
import sys, math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    if len(toks) < 5:
        fail("bad instance header")
    N = int(toks[0]); cx = float(toks[1]); cy = float(toks[2])
    R = float(toks[3]); K = int(toks[4])
    zones = []
    idx = 5
    for _ in range(K):
        zx = float(toks[idx]); zy = float(toks[idx + 1]); zr = float(toks[idx + 2])
        idx += 3
        zones.append((zx, zy, zr))
    return N, cx, cy, R, zones


# ---- checker's internal baseline: airspace-aware equal-radius grid (must match trivial.py) ----
def baseline_disks(N, cx, cy, R, zones):
    gx = int(math.ceil(math.sqrt(N)))
    if gx < 1:
        gx = 1
    gy = int(math.ceil(N / float(gx)))
    cw = (2.0 * R) / gx
    ch = (2.0 * R) / gy
    r = min(cw, ch) * 0.5 * 0.999
    x0 = cx - R
    y0 = cy - R
    disks = []
    for j in range(gy):
        for i in range(gx):
            if len(disks) >= N:
                break
            px = x0 + (i + 0.5) * cw
            py = y0 + (j + 0.5) * ch
            # containment in airspace disk
            if math.hypot(px - cx, py - cy) + r > R:
                continue
            # no-fly clearance
            ok = True
            for (zx, zy, zr) in zones:
                if math.hypot(px - zx, py - zy) < r + zr:
                    ok = False
                    break
            if not ok:
                continue
            # non-overlap with placed
            for (ox, oy, orr) in disks:
                if math.hypot(px - ox, py - oy) < r + orr - 1e-12:
                    ok = False
                    break
            if ok:
                disks.append((px, py, r))
    return disks


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, cx, cy, R, zones = read_instance(in_path)

    try:
        with open(out_path) as f:
            data = f.read().split()
    except Exception:
        fail("no output")
    if len(data) < 1:
        fail("empty output")
    try:
        M = int(data[0])
    except Exception:
        fail("first token not integer M")
    if M < 0 or M > N:
        fail("M out of range")
    if len(data) < 1 + 3 * M:
        fail("not enough coordinates for M bubbles")

    disks = []
    for k in range(M):
        try:
            x = float(data[1 + 3 * k])
            y = float(data[2 + 3 * k])
            r = float(data[3 + 3 * k])
        except Exception:
            fail("non-numeric bubble field")
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            fail("non-finite value")
        if r < -TOL:
            fail("negative radius")
        if r < 0:
            r = 0.0
        # containment in circular airspace
        if math.hypot(x - cx, y - cy) + r > R + TOL:
            fail("bubble leaves the airspace")
        # no-fly clearance
        for (zx, zy, zr) in zones:
            if math.hypot(x - zx, y - zy) < r + zr - TOL:
                fail("bubble enters a no-fly zone")
        disks.append((x, y, r))

    # pairwise non-overlap
    for i in range(M):
        xi, yi, ri = disks[i]
        for j in range(i + 1, M):
            xj, yj, rj = disks[j]
            if math.hypot(xi - xj, yi - yj) < ri + rj - TOL:
                fail("two drone bubbles overlap")

    F = sum(d[2] for d in disks)

    base = baseline_disks(N, cx, cy, R, zones)
    B = sum(d[2] for d in base)
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
