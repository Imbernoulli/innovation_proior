# TIER: strong
"""
Insight: no single regime's narrow window has enough curvature to pin down
the shared law on its own -- locally, sin(w*x+phi) is nearly degenerate with
many other (w, phase, linear-drift) combinations. But if you DIVIDE OUT each
regime's own (gain_r, offset_r) -- exactly the ratio/difference trick the
statement describes -- and pool the nuisance-corrected residuals from every
regime together, the pooled points span the FULL x-domain (all five narrow
windows at once), which is wide enough to identify the shared oscillation +
drift unambiguously. This turns single-window curve-fitting (underdetermined)
into a cross-regime identification problem (well determined).

Algorithm, for each candidate shared frequency w on a grid:
  - alternate between (a) given the current shared shape, refit each
    regime's own (gain_r, offset_r) by a simple 2-parameter OLS against that
    regime's own rows only, and (b) given the current per-regime
    (gain_r, offset_r), pool EVERY regime's nuisance-corrected points
    (y - offset_r) / gain_r together and re-fit the shared
    P*sin(wx) + Q*cos(wx) + c*x by ordinary least squares over the full pool;
  - after a few rounds this converges; keep whichever w in the grid gives the
    lowest total reconstruction error across all regimes.
Read off phi = atan2(Q, P) after normalising (P, Q) to unit length, and
rescale c by the same factor, since the sinusoid's amplitude is fixed to 1 in
the true law (any pure overall scale is nuisance the grader re-fits anyway).
"""
import sys, math


def solve_n(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            M[piv][c] += 1e-9
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r != c:
                f = M[r][c]
                if f != 0.0:
                    for j in range(c, n + 1):
                        M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def regime_affine(core_hat, ys):
    n = len(ys)
    mc = sum(core_hat) / n
    my = sum(ys) / n
    sxy = sum((ch - mc) * (y - my) for ch, y in zip(core_hat, ys))
    sxx = sum((ch - mc) ** 2 for ch in core_hat)
    if sxx < 1e-9:
        return 0.0, my
    g = sxy / sxx
    return g, my - g * mc


def _alternate_from(regimes, w, P, Q, c, iters=10):
    """Alternating least squares started from a given shared-shape guess:
    (a) refit each regime's own (gain_r, offset_r) against the current
    shape, (b) pool every regime's nuisance-corrected points and refit the
    shared (P, Q, c) over the FULL pooled x-range (well conditioned, unlike
    any single narrow window)."""
    gains = offsets = None
    for _ in range(iters):
        gains, offsets = [], []
        for xs, ys in regimes:
            core_hat = [P * math.sin(w * x) + Q * math.cos(w * x) + c * x for x in xs]
            g, o = regime_affine(core_hat, ys)
            gains.append(g)
            offsets.append(o)

        AtA = [[0.0] * 3 for _ in range(3)]
        Atb = [0.0] * 3
        for (xs, ys), g, o in zip(regimes, gains, offsets):
            if abs(g) < 1e-6:
                continue
            for x, y in zip(xs, ys):
                z = (y - o) / g
                f = [math.sin(w * x), math.cos(w * x), x]
                for i in range(3):
                    Atb[i] += f[i] * z
                    for j in range(3):
                        AtA[i][j] += f[i] * f[j]
        for i in range(3):
            AtA[i][i] += 1e-6
        P, Q, c = solve_n(AtA, Atb)

    sse = 0.0
    for (xs, ys), g, o in zip(regimes, gains, offsets):
        for x, y in zip(xs, ys):
            ch = P * math.sin(w * x) + Q * math.cos(w * x) + c * x
            sse += (g * ch + o - y) ** 2
    return P, Q, c, sse


# a fixed spread of starting phases/drifts -- the alternating fit is a
# biconvex problem with more than one basin (in particular c=0 is a shallow
# trap), so try several starts and keep the best.
_START_PHASES = [i * 2 * math.pi / 8 for i in range(8)]
_START_C = [0.0, 0.15, -0.15]


def fit_shared_shape(regimes, w):
    best = None
    for phi0 in _START_PHASES:
        for c0 in _START_C:
            P, Q, c, sse = _alternate_from(regimes, w, math.cos(phi0), math.sin(phi0), c0)
            if best is None or sse < best[3]:
                best = (P, Q, c, sse)
    return best


def search(regimes, lo, hi, step):
    best = None
    w = lo
    while w <= hi + 1e-9:
        P, Q, c, sse = fit_shared_shape(regimes, w)
        if best is None or sse < best[0]:
            best = (sse, w, P, Q, c)
        w += step
    return best


def main():
    data = sys.stdin.read().split()
    idx = 0
    K = int(data[idx]); idx += 1
    idx += 1  # test id
    regimes = []
    for _ in range(K):
        idx += 1  # regime id
        n = int(data[idx]); idx += 1
        idx += 1  # lo
        idx += 1  # hi
        xs, ys = [], []
        for _ in range(n):
            x = float(data[idx]); idx += 1
            y = float(data[idx]); idx += 1
            xs.append(x); ys.append(y)
        regimes.append((xs, ys))

    coarse = search(regimes, 0.75, 1.95, 0.03)
    _, w0, _, _, _ = coarse
    fine = search(regimes, max(0.70, w0 - 0.03), min(2.00, w0 + 0.03), 0.003)
    _, w, P0, Q0, c0 = fine

    amp = math.sqrt(P0 * P0 + Q0 * Q0)
    if amp < 1e-9:
        print("0")
        return
    P, Q = P0 / amp, Q0 / amp
    c = c0 / amp
    phi = math.atan2(Q, P)

    print("sin(%.6f*x + %.6f) + %.6f*x" % (w, phi, c))


if __name__ == "__main__":
    main()
