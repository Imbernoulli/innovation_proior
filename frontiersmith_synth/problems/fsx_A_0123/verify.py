# verify.py -- deterministic scorer for the data-center cooling diffuser-packing problem.
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints exactly one "Ratio: <float in [0,1]>" line and exits 0.
#
# Feasibility (tol = 1e-6): 0<=M<=N; each r>=-tol and finite; each diffuser disk lies
# inside the room [0,W]x[0,H]; disks are pairwise non-overlapping; and no disk intersects
# any rack keep-out rectangle. Objective F = sum of radii (maximize). Ratio = min(1, 0.1*F/B)
# where B is the checker's own obstacle-aware equal-radius grid baseline.
import sys, math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    if len(toks) < 4:
        fail("bad instance header")
    N = int(toks[0]); W = float(toks[1]); H = float(toks[2]); K = int(toks[3])
    racks = []
    idx = 4
    for _ in range(K):
        x0 = float(toks[idx]); y0 = float(toks[idx + 1])
        x1 = float(toks[idx + 2]); y1 = float(toks[idx + 3])
        idx += 4
        racks.append((x0, y0, x1, y1))
    return N, W, H, racks


def dist_pt_rect(px, py, rect):
    x0, y0, x1, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


# ---- checker's internal baseline: obstacle-aware single centre row (must match trivial.py) ----
def baseline_disks(N, W, H, racks):
    r = min(W / (2.0 * N), H * 0.5) * 0.999
    cy = H * 0.5
    disks = []
    for i in range(N):
        cx = (i + 0.5) * W / N
        if cx - r < 0 or cx + r > W or cy - r < 0 or cy + r > H:
            continue
        ok = True
        for rk in racks:
            if dist_pt_rect(cx, cy, rk) < r:
                ok = False
                break
        if ok:
            disks.append((cx, cy, r))
    return disks


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, W, H, racks = read_instance(in_path)

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
        fail("not enough coordinates for M disks")

    disks = []
    for k in range(M):
        try:
            x = float(data[1 + 3 * k])
            y = float(data[2 + 3 * k])
            r = float(data[3 + 3 * k])
        except Exception:
            fail("non-numeric disk field")
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            fail("non-finite value")
        if r < -TOL:
            fail("negative radius")
        if r < 0:
            r = 0.0
        # containment in room
        if x - r < -TOL or x + r > W + TOL or y - r < -TOL or y + r > H + TOL:
            fail("disk leaves the room")
        # rack clearance
        for rk in racks:
            if dist_pt_rect(x, y, rk) < r - TOL:
                fail("disk intersects a server rack")
        disks.append((x, y, r))

    # pairwise non-overlap
    for i in range(M):
        xi, yi, ri = disks[i]
        for j in range(i + 1, M):
            xj, yj, rj = disks[j]
            if math.hypot(xi - xj, yi - yj) < ri + rj - TOL:
                fail("two diffuser disks overlap")

    F = sum(d[2] for d in disks)

    base = baseline_disks(N, W, H, racks)
    B = sum(d[2] for d in base)
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
