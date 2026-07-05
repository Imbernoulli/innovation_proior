# verify.py -- deterministic scorer for the Rotunda Gallery Tour packing problem.
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints exactly one "Ratio: <float in [0,1]>" line and exits 0.
import sys, math

TOL = 1e-6

def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)

def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    with open(in_path) as f:
        toks = f.read().split()
    if len(toks) < 3:
        fail("bad instance")
    N = int(toks[0]); R = float(toks[1]); r_in = float(toks[2])

    # ---- read participant artifact ----
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
    need = 1 + 3 * M
    if len(data) < need:
        fail("not enough coordinates for M zones")

    disks = []
    idx = 1
    for _ in range(M):
        try:
            x = float(data[idx]); y = float(data[idx + 1]); r = float(data[idx + 2])
        except Exception:
            fail("non-numeric coordinate")
        idx += 3
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            fail("non-finite value")
        if r < -TOL:
            fail("negative radius")
        disks.append((x, y, max(0.0, r)))

    # ---- feasibility ----
    for (x, y, r) in disks:
        d = math.hypot(x, y)
        if d + r > R + TOL:
            fail("zone leaves outer wall")
        if d - r < r_in - TOL:
            fail("zone crosses central cordon")
    for i in range(M):
        xi, yi, ri = disks[i]
        for j in range(i + 1, M):
            xj, yj, rj = disks[j]
            if math.hypot(xi - xj, yi - yj) < ri + rj - TOL:
                fail("zones overlap")

    F = sum(r for (_, _, r) in disks)

    # ---- internal baseline B: single ring on the mid-annulus circle ----
    rmid = (r_in + R) / 2.0
    w = R - r_in
    if N >= 1:
        r_ang = rmid * math.sin(math.pi / N) if N >= 2 else w / 2.0
        r_base = min(w / 2.0, r_ang)
        B = N * r_base
    else:
        B = 1e-9
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * F / B)
    print("Ratio: %.6f" % (sc / 1000.0))

if __name__ == "__main__":
    main()
