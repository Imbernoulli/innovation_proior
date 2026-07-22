# TIER: strong
"""Insight: the color-dependent bending is not noise to average away -- it is
the ONLY lever that can route three targets from one shared entry point,
because a single lattice feature bends each color by a DIFFERENT amount
(chromatic-splitting). Rather than aiming a lens at one reference color,
search the lens (center, width, strength, extent) to jointly maximize the
TOTAL score across all three colors at once -- deliberately trading a worse
fit for the reference color for a much better joint placement that the
per-color-blind greedy lens can never reach. This is a genuine reformulation
(a joint/coupled objective) rather than "greedy plus more iterations".
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
        r = int(clampf(int(math.floor(y)), 0, H - 1))
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


def total_F(L, W, H, rays, CHROMA, ALPHA, KAPPA, T):
    F = 0.0
    for c in range(3):
        for y0 in rays[c]:
            ye = trace_ray(L, W, H, y0, CHROMA[c], ALPHA, KAPPA)
            F += 1.0 / (1.0 + abs(ye - T[c]))
    return F


def bump(W, H, LMAX, center, sigma, amp, x1):
    L = [[0] * H for _ in range(W)]
    x1 = max(1, min(W, x1))
    if sigma <= 0:
        sigma = 0.5
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

    # coarse joint grid search over the lens family, maximizing the coupled
    # 3-color objective (not any single color's fit)
    stride = max(1, H // 14)
    centers = list(range(0, H, stride))
    sigmas = (1, 2, 3, 4, 6, 8)
    amps = list(range(1, LMAX + 1))
    extents = sorted({max(1, W // 4), max(1, W // 2), max(1, 3 * W // 4), W})

    best = None
    for center in centers:
        for sigma in sigmas:
            for amp in amps:
                for x1 in extents:
                    L = bump(W, H, LMAX, center, sigma, amp, x1)
                    if budget(L) > K:
                        continue
                    F = total_F(L, W, H, rays, CHROMA, ALPHA, KAPPA, T)
                    if best is None or F > best[0]:
                        best = (F, center, sigma, amp, x1)

    # local refinement around the best coarse candidate
    if best is not None:
        F0, c0, s0, a0, x0 = best
        for center in range(max(0, c0 - stride), min(H, c0 + stride + 1)):
            for sigma in (s0 - 0.5, s0, s0 + 0.5, s0 + 1.0):
                if sigma <= 0:
                    continue
                for amp in {max(1, a0 - 1), a0, min(LMAX, a0 + 1)}:
                    for x1 in {max(1, x0 - 2), x0, min(W, x0 + 2)}:
                        L = bump(W, H, LMAX, center, sigma, amp, x1)
                        if budget(L) > K:
                            continue
                        F = total_F(L, W, H, rays, CHROMA, ALPHA, KAPPA, T)
                        if F > best[0]:
                            best = (F, center, sigma, amp, x1)

    if best is None:
        L = [[0] * H for _ in range(W)]
    else:
        L = bump(W, H, LMAX, best[1], best[2], best[3], best[4])

    lines = []
    for y in range(H):
        lines.append(" ".join(str(L[x][y]) for x in range(W)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
