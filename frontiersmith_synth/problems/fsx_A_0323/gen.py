import sys

# Highway Toll Gantries -- deterministic instance generator.
# `python3 gen.py <testId>`  (testId 1..10 = difficulty ladder, small -> large).
# The highway is the strip [0, L] x [0, W]; W = 1.0, L = 0.15 * N (a long thin road).
# There are K fixed "pylon" obstacles (small disks of radius rho) confined to the two
# EDGE bands (near y=0 and y=W); the central corridor is always kept clear so a single
# center row of sensor disks is always feasible.  Everything is seeded by testId only.

# Large-scale ladder: number of gantry sensor disks grows.
LADDER = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120]

W = 1.0
RHO = 0.06

def lcg(seed):
    # deterministic 32-bit LCG (Numerical Recipes constants)
    s = seed & 0xFFFFFFFF
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s / 4294967296.0

def main():
    tid = int(sys.argv[1])
    idx = min(max(tid, 1), len(LADDER)) - 1
    N = LADDER[idx]
    L = 0.15 * N
    K = max(3, N // 8)

    rng = lcg(97 * tid + 12345)
    # bottom band: y in [RHO, W/4 - RHO]; top band: y in [3W/4 + RHO, W - RHO]
    b_lo, b_hi = RHO, W / 4.0 - RHO
    t_lo, t_hi = 3.0 * W / 4.0 + RHO, W - RHO

    pylons = []
    for j in range(K):
        u = next(rng); v = next(rng); w = next(rng)
        x = RHO + u * (L - 2.0 * RHO)
        if w < 0.5:
            y = b_lo + v * (b_hi - b_lo)
        else:
            y = t_lo + v * (t_hi - t_lo)
        pylons.append((x, y))

    out = ["%d %.6f %.6f %d %.6f" % (N, L, W, K, RHO)]
    for (x, y) in pylons:
        out.append("%.6f %.6f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
