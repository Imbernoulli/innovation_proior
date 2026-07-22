#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for refraction-lattice-lensing.

Reads the instance from <in>, the participant's refractive-level grid from <out>,
simulates each ray with the SAME physics described in statement.md, and prints
`Ratio: <float in [0,1]>` on its own final line, then exits 0.
"""
import sys, math

MAXSLOPE = 4.0


def clampf(v, lo, hi):
    return max(lo, min(hi, v))


def idx(level, chroma, alpha):
    return 1.0 + alpha * chroma * level


def trace_ray(Lvl, W, H, y0, chroma, alpha, kappa):
    """Lvl[x][y] integer level field. Returns the ray's exit row (real) at column W-1."""
    y = float(y0)
    s = 0.0
    for x in range(W - 1):
        r = int(math.floor(y))
        if r < 0: r = 0
        if r > H - 1: r = H - 1
        ru = r + 1 if r + 1 <= H - 1 else H - 1
        rd = r - 1 if r - 1 >= 0 else 0
        # gradient-index-design: continuous transverse bending from the local
        # row-gradient of the (color-dependent) refractive index.
        grad = idx(Lvl[x][ru], chroma, alpha) - idx(Lvl[x][rd], chroma, alpha)
        s = s + kappa * grad
        y_prov = y + s
        r2 = int(math.floor(y_prov))
        if r2 < 0: r2 = 0
        if r2 > H - 1: r2 = H - 1
        if r2 != r:
            # discrete-snell-steering: crossing into a new row-cell refracts the
            # ray, conserving the tangential invariant n / sqrt(1+s^2).
            n1 = idx(Lvl[x][r], chroma, alpha)
            n2 = idx(Lvl[x][r2], chroma, alpha)
            denom = math.sqrt(1.0 + s * s)
            rhs = (n1 / n2) / denom if n2 > 1e-12 else 1.0
            rhs = clampf(rhs, 1e-9, 1.0 - 1e-9)
            sign = 1.0 if s >= 0.0 else -1.0
            s = sign * math.sqrt(max(0.0, 1.0 / (rhs * rhs) - 1.0))
        y = y + s
        s = clampf(s, -MAXSLOPE, MAXSLOPE)
        y = clampf(y, 0.0, H - 1e-9)
    return y


def total_F(Lvl, W, H, rays, chromas, alpha, kappa, targets):
    F = 0.0
    for c in range(3):
        for y0 in rays[c]:
            ye = trace_ray(Lvl, W, H, y0, chromas[c], alpha, kappa)
            F += 1.0 / (1.0 + abs(ye - targets[c]))
    return F


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        itoks = f.read().split()
    idx_i = 0
    def nexti():
        nonlocal idx_i
        v = itoks[idx_i]; idx_i += 1
        return v

    W = int(nexti()); H = int(nexti())
    LMAX = int(nexti()); K = int(nexti())
    ALPHA = float(nexti()); KAPPA = float(nexti())
    CHROMA = [float(nexti()) for _ in range(3)]
    T = [int(nexti()) for _ in range(3)]
    R = int(nexti())
    rays = []
    for c in range(3):
        rays.append([int(nexti()) for _ in range(R)])

    # ---- read participant output strictly ----
    try:
        with open(outf) as f:
            otoks = f.read().split()
    except Exception as e:
        fail("cannot read output: %s" % e)

    if len(otoks) != W * H:
        fail("expected exactly %d tokens (H=%d rows x W=%d cols), got %d" % (W * H, H, W, len(otoks)))

    Lvl_rowmajor = []  # Lvl_rowmajor[y][x]
    total = 0
    for tok in otoks:
        # strict integer literal check (rejects floats, nan, inf, garbage)
        s = tok
        neg = s.startswith("-")
        body = s[1:] if neg else s
        if body == "" or not body.isdigit():
            fail("non-integer token %r" % tok)
        v = int(s)
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite token %r" % tok)
        if v < 0 or v > LMAX:
            fail("level %d out of range [0,%d]" % (v, LMAX))
        total += v
        Lvl_rowmajor.append(v)

    if total > K:
        fail("budget exceeded: used %d > K=%d" % (total, K))

    # reshape to Lvl[x][y]
    Lvl = [[0] * H for _ in range(W)]
    p = 0
    for y in range(H):
        for x in range(W):
            Lvl[x][y] = Lvl_rowmajor[p]
            p += 1

    F = total_F(Lvl, W, H, rays, CHROMA, ALPHA, KAPPA, T)

    # internal trivial baseline: the all-zero lattice (no budget spent at all)
    Lvl0 = [[0] * H for _ in range(W)]
    B = total_F(Lvl0, W, H, rays, CHROMA, ALPHA, KAPPA, T)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f budget_used=%d/%d" % (F, B, total, K))
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
