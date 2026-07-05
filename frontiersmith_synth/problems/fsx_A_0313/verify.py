#!/usr/bin/env python3
# Deterministic checker for the Solar-Farm Inverter Clearance Packing problem
# (format C, maximize the sum of clearance-circle radii inside a rectangular plot
# with fixed circular keep-out obstacles).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0, 1].
import sys
import math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    p = 0
    N = int(toks[p]); p += 1
    W = float(toks[p]); p += 1
    H = float(toks[p]); p += 1
    M = int(toks[p]); p += 1
    obs = []
    for _ in range(M):
        ox = float(toks[p]); p += 1
        oy = float(toks[p]); p += 1
        oR = float(toks[p]); p += 1
        obs.append((ox, oy, oR))
    return N, W, H, obs


def baseline(N, W, H, obs):
    """Trivial feasible reference: a coarse row-major grid of small equal-ish
    circles, each shrunk to respect the plot walls and the keep-out obstacles.
    Returns the list of (x, y, r) circles; the baseline value B is their sum."""
    gc = int(math.ceil(math.sqrt(N)))
    gr = int(math.ceil(float(N) / gc))
    cw = W / gc
    ch = H / gr
    frac = 0.30
    circles = []
    count = 0
    for rr in range(gr):
        for cc in range(gc):
            if count >= N:
                break
            cx = (cc + 0.5) * cw
            cy = (rr + 0.5) * ch
            rad = frac * min(cw, ch)
            rad = min(rad, cx, W - cx, cy, H - cy)
            for (ox, oy, oR) in obs:
                d = math.hypot(cx - ox, cy - oy)
                rad = min(rad, d - oR)
            if rad > 0.0:
                circles.append((cx, cy, rad))
            count += 1
    return circles


def main():
    try:
        N, W, H, obs = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if not otoks:
        fail("empty output")

    try:
        K = int(otoks[0])
    except Exception:
        fail("bad K")

    if K < 0 or K > N:
        fail("K out of range")

    need = 1 + 3 * K
    if len(otoks) < need:
        fail("truncated circles")

    xs = []
    ys = []
    rs = []
    for k in range(K):
        try:
            x = float(otoks[1 + 3 * k])
            y = float(otoks[2 + 3 * k])
            r = float(otoks[3 + 3 * k])
        except Exception:
            fail("bad circle %d" % k)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            fail("non-finite circle %d" % k)
        if r < -TOL:
            fail("negative radius %d" % k)
        # containment in the rectangular plot [0,W] x [0,H]
        if x - r < -TOL or x + r > W + TOL or y - r < -TOL or y + r > H + TOL:
            fail("circle %d outside plot" % k)
        # clear of every fixed keep-out obstacle
        for (ox, oy, oR) in obs:
            d = math.hypot(x - ox, y - oy)
            if d < r + oR - TOL:
                fail("circle %d hits keep-out" % k)
        xs.append(x)
        ys.append(y)
        rs.append(r)

    # pairwise non-overlap (O(K^2), K small)
    for a in range(K):
        for b in range(a + 1, K):
            dx = xs[a] - xs[b]
            dy = ys[a] - ys[b]
            d = math.sqrt(dx * dx + dy * dy)
            if d < rs[a] + rs[b] - TOL:
                fail("overlap %d,%d" % (a, b))

    F = sum(rs)

    base = baseline(N, W, H, obs)
    B = sum(c[2] for c in base)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
