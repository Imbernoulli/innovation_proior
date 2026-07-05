# TIER: strong
# Discover the correct Ludwik + rate-hardening form  sigma = Y + K*s**n + C*r**q.
# The hardening exponent n and rate exponent q are selected by an INTERNAL train/validation
# split (hold out the largest-strain and largest-rate training rows) so the fit is chosen for
# EXTRAPOLATION rather than in-sample noise; then (Y, K, C) are refit on the full table by
# linear least squares given the chosen exponents.
import sys, math


def lstsq(rows, y):
    m = len(rows[0])
    A = [[0.0] * m for _ in range(m)]
    bvec = [0.0] * m
    for r, yy in zip(rows, y):
        for i in range(m):
            bvec[i] += r[i] * yy
            for j in range(m):
                A[i][j] += r[i] * r[j]
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


def rmse(pred, obs):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, obs)) / len(obs))


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    idx = 1
    pts = []
    for _ in range(n):
        s = float(toks[idx]); r = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        pts.append((s, r, d))

    smax = max(s for s, _, _ in pts)
    rmax = max(r for _, r, _ in pts)
    # inner validation = the outer edge of the training envelope (largest strain / rate rows)
    val = [p for p in pts if p[0] >= smax - 1e-9 or p[1] >= rmax - 1e-9]
    fit = [p for p in pts if not (p[0] >= smax - 1e-9 or p[1] >= rmax - 1e-9)]

    best = None
    ni = 6
    while ni <= 14:               # n in {0.30 .. 0.70}
        nexp = ni * 0.05
        qi = 4
        while qi <= 10:           # q in {0.04 .. 0.10} scaled below
            qexp = qi * 0.02
            rows = [[1.0, s ** nexp, r ** qexp] for s, r, _ in fit]
            co = lstsq(rows, [d for _, _, d in fit])
            if co is not None:
                pr = [co[0] + co[1] * s ** nexp + co[2] * r ** qexp for s, r, _ in val]
                e = rmse(pr, [d for _, _, d in val])
                if best is None or e < best[0]:
                    best = (e, nexp, qexp)
            qi += 1
        ni += 1

    _, nexp, qexp = best
    rows = [[1.0, s ** nexp, r ** qexp] for s, r, _ in pts]
    Y, K, C = lstsq(rows, [d for _, _, d in pts])
    print("%.10g + %.10g * s ** %.10g + %.10g * r ** %.10g"
          % (Y, K, nexp, C, qexp))


if __name__ == "__main__":
    main()
