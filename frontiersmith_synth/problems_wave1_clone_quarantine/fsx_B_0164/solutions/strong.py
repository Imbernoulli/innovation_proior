# TIER: strong
# Discover the correct additive scaling form  D = E + A*T**(-alpha) + B*V**beta.
# Exponents (alpha, beta) are chosen by an INTERNAL train/validation split (hold out the
# largest-T and largest-V training rows) so the fit is selected for EXTRAPOLATION rather
# than in-sample noise; then (E, A, B) are refit on the full table by linear least squares.
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
        t = float(toks[idx]); v = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        pts.append((t, v, d))

    tmax = max(t for t, _, _ in pts)
    vmax = max(v for _, v, _ in pts)
    val = [r for r in pts if r[0] >= tmax - 0.5 or r[1] >= vmax - 0.5]
    fit = [r for r in pts if not (r[0] >= tmax - 0.5 or r[1] >= vmax - 0.5)]

    best = None
    ai = 10
    while ai <= 20:
        alpha = ai * 0.05
        bi = 20
        while bi <= 40:
            beta = bi * 0.05
            rows = [[1.0, t ** (-alpha), v ** beta] for t, v, _ in fit]
            co = lstsq(rows, [d for _, _, d in fit])
            if co is not None:
                pr = [co[0] + co[1] * t ** (-alpha) + co[2] * v ** beta for t, v, _ in val]
                e = rmse(pr, [d for _, _, d in val])
                if best is None or e < best[0]:
                    best = (e, alpha, beta)
            bi += 1
        ai += 1

    _, alpha, beta = best
    rows = [[1.0, t ** (-alpha), v ** beta] for t, v, _ in pts]
    E, A, B = lstsq(rows, [d for _, _, d in pts])
    print("%.10g + %.10g * T ** %.10g + %.10g * V ** %.10g"
          % (E, A, -alpha, B, beta))


if __name__ == "__main__":
    main()
