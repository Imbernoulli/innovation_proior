# TIER: strong
# Discover the correct COMBINED-RESOURCE super-linear scaling form
#     D = D0 + C * (A + rho*H) ** q         (q > 1)
# The nonlinear knobs (rho, q) are chosen by an INTERNAL train/validation split (hold out
# the largest-area and hottest training rows) so the exponent is selected for EXTRAPOLATION
# rather than in-sample noise; then (D0, C) are refit on the full table by least squares.
# Fits use relative (multiplicative-noise) weighting so large-D rows don't swamp the fit.
# Irreducible measurement noise keeps the held-out error above zero.
import sys, math


def lstsq(rows, y, wts=None):
    m = len(rows[0])
    if wts is None:
        wts = [1.0] * len(rows)
    A = [[0.0] * m for _ in range(m)]
    bvec = [0.0] * m
    for r, yy, wv in zip(rows, y, wts):
        for i in range(m):
            bvec[i] += wv * r[i] * yy
            for j in range(m):
                A[i][j] += wv * r[i] * r[j]
    M = [A[i][:] + [bvec[i]] for i in range(m)]
    for c in range(m):
        piv = max(range(c, m), key=lambda rr: abs(M[rr][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(m):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][m] / M[i][i] for i in range(m)]


def rel_rmse(pred, obs):
    return math.sqrt(sum(((a - b) / b) ** 2 for a, b in zip(pred, obs)) / len(obs))


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    idx = 1
    pts = []
    for _ in range(n):
        a = float(toks[idx]); h = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        pts.append((a, h, d))

    amax = max(a for a, _, _ in pts)
    hmax = max(h for _, h, _ in pts)
    val = [r for r in pts if r[0] >= amax - 0.5 or r[1] >= hmax - 0.5]
    fit = [r for r in pts if not (r[0] >= amax - 0.5 or r[1] >= hmax - 0.5)]

    best = None
    ri = 6
    while ri <= 24:                       # rho in 0.6 .. 2.4
        rho = ri * 0.1
        qi = 20
        while qi <= 36:                   # q in 1.00 .. 1.80
            q = qi * 0.05
            rows = [[1.0, (a + rho * h) ** q] for a, h, _ in fit]
            wts = [1.0 / (d * d) for _, _, d in fit]
            co = lstsq(rows, [d for _, _, d in fit], wts)
            if co is not None:
                pr = [co[0] + co[1] * (a + rho * h) ** q for a, h, _ in val]
                e = rel_rmse(pr, [d for _, _, d in val])
                if best is None or e < best[0]:
                    best = (e, rho, q)
            qi += 1
        ri += 1

    _, rho, q = best
    rows = [[1.0, (a + rho * h) ** q] for a, h, _ in pts]
    wts = [1.0 / (d * d) for _, _, d in pts]
    D0, C = lstsq(rows, [d for _, _, d in pts], wts)
    print("%.10g + %.10g * (A + %.10g * H) ** %.10g" % (D0, C, rho, q))


if __name__ == "__main__":
    main()
