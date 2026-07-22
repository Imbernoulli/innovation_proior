# TIER: greedy
"""The obvious textbook approach: build ONE symmetric converging lens (a
Gaussian bump in refractive level, centered where the beam enters) and tune
its strength/width/extent to focus the REFERENCE color (the middle-dispersion
color) onto its own target -- treating the other two colors' different
bending as noise to ignore, not as a signal to exploit. This is exactly the
classical-lens trap: it nails one color and disperses the other two.
"""
import sys, math

MAXSLOPE = 4.0


def clampf(v, lo, hi):
    return max(lo, min(hi, v))


def idx(level, chroma, alpha):
    return 1.0 + alpha * chroma * level


def trace_ray(Lvl, W, H, y0, chroma, alpha, kappa):
    y = float(y0); s = 0.0
    for x in range(W - 1):
        r = clampf(int(math.floor(y)), 0, H - 1)
        r = int(r)
        ru = min(r + 1, H - 1); rd = max(r - 1, 0)
        grad = idx(Lvl[x][ru], chroma, alpha) - idx(Lvl[x][rd], chroma, alpha)
        s = s + kappa * grad
        y_prov = y + s
        r2 = int(clampf(int(math.floor(y_prov)), 0, H - 1))
        if r2 != r:
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


def bump(W, H, LMAX, center, sigma, amp, x1):
    L = [[0] * H for _ in range(W)]
    x1 = max(1, min(W, x1))
    for x in range(x1):
        for y in range(H):
            v = amp * math.exp(-((y - center) ** 2) / (2.0 * sigma * sigma))
            lv = int(round(v))
            if lv < 0: lv = 0
            if lv > LMAX: lv = LMAX
            L[x][y] = lv
    return L


def budget(L):
    return sum(sum(row) for row in L)


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    def nx(): return next(it)
    W = int(nx()); H = int(nx())
    LMAX = int(nx()); K = int(nx())
    ALPHA = float(nx()); KAPPA = float(nx())
    CHROMA = [float(nx()) for _ in range(3)]
    T = [int(nx()) for _ in range(3)]
    R = int(nx())
    rays = [[int(nx()) for _ in range(R)] for _ in range(3)]

    ref = 1  # the middle-dispersion color: treat it as "the" target
    entry = rays[ref][len(rays[ref]) // 2]

    best = None
    for sigma in (1, 2, 3, 4, 6, 8, 12):
        for amp in range(1, LMAX + 1):
            for x1 in (max(1, W // 4), max(1, W // 2), max(1, 3 * W // 4), W):
                L = bump(W, H, LMAX, entry, sigma, amp, x1)
                if budget(L) > K:
                    continue
                sc = 0.0
                for y0 in rays[ref]:
                    ye = trace_ray(L, W, H, y0, CHROMA[ref], ALPHA, KAPPA)
                    sc += 1.0 / (1.0 + abs(ye - T[ref]))
                if best is None or sc > best[0]:
                    best = (sc, sigma, amp, x1)

    if best is None:
        L = [[0] * H for _ in range(W)]
    else:
        L = bump(W, H, LMAX, entry, best[1], best[2], best[3])

    lines = []
    for y in range(H):
        lines.append(" ".join(str(L[x][y]) for x in range(W)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
