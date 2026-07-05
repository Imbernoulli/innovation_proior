# TIER: strong
# Structured symbolic search for the asymmetric outbreak wave.
# Guess the mechanistic feature family
#     curve(t) = 1 / ( exp(-a*(x1-tp)) + exp(b*(x1-tp)) )
# grid-search the hidden rise rate a, decay rate b and peak week tp, and for
# each candidate solve the LINEAR coefficients [A, d, e, g] of
#     y ~ A*curve + d*x2 + e*x3 + g
# by least squares, keeping the best training fit.  This recovers the shape of
# the hidden incidence law (crucially its slow exponential tail), so it
# extrapolates into the post-peak window far better than a polynomial.  The
# coarse rate grid + only glimpsing the decline in-window leave headroom
# below 1.0.
import sys, math


def lstsq(A, y):
    m = len(A); n = len(A[0])
    M = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(m):
        for j in range(n):
            M[j][n] += A[i][j] * y[i]
            for k in range(n):
                M[j][k] += A[i][j] * A[i][k]
    for j in range(n):
        M[j][j] += 1e-6
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col] or 1e-12
        for k in range(col, n + 1):
            M[col][k] /= pv
        for r in range(n):
            if r != col:
                f = M[r][col]
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    return [M[j][n] for j in range(n)]


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    X = []; y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 5:
            v = list(map(float, p[:5]))
            X.append(v[:4]); y.append(v[4])

    A_grid = [0.55, 0.70, 0.85]
    b_grid = [0.30, 0.38, 0.46]
    tp_grid = [6.0, 6.7, 7.4, 8.0]
    best = None
    for ag in A_grid:
        for bg in b_grid:
            for tg in tp_grid:
                F = []
                for x in X:
                    u = x[0] - tg
                    cv = 1.0 / (math.exp(-ag * u) + math.exp(bg * u))
                    F.append([cv, x[1], x[2], 1.0])
                cf = lstsq(F, y)
                res = 0.0
                for x, yy in zip(X, y):
                    u = x[0] - tg
                    cv = 1.0 / (math.exp(-ag * u) + math.exp(bg * u))
                    pr = cf[0] * cv + cf[1] * x[1] + cf[2] * x[2] + cf[3]
                    res += (pr - yy) ** 2
                if best is None or res < best[0]:
                    best = (res, ag, bg, tg, cf)
    _, ag, bg, tg, cf = best
    print("%r / (exp(%r * (x1 - %r)) + exp(%r * (x1 - %r))) + %r * x2 + %r * x3 + %r"
          % (cf[0], -ag, tg, bg, tg, cf[1], cf[2], cf[3]))


if __name__ == "__main__":
    main()
