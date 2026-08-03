# TIER: strong
# Structured symbolic search: guess the feature family
#     [1, x3*exp(-c*x4), x1*x2, x2^2, x1]
# and grid-search the hidden staleness-decay rate c, picking the best training
# fit, then solve the linear coefficients by least squares.  Recovers the shape
# of the hidden fresh-yield law so it extrapolates far better than the linear
# fit, but the coarse c-grid + irreducible logging noise leave headroom < 1.
import sys, math


def lstsq(A, y):
    m = len(A); n = len(A[0])
    M = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(m):
        for j in range(n):
            M[j][n] += A[i][j] * y[i]
            for k in range(n):
                M[j][k] += A[i][j] * A[i][k]
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
    best = None
    for cg in [0.55, 0.7, 0.85, 1.0, 1.15]:
        A = [[1.0, x[2] * math.exp(-cg * x[3]), x[0] * x[1], x[1] * x[1], x[0]] for x in X]
        b = lstsq(A, y)
        res = 0.0
        for x, yy in zip(X, y):
            pr = (b[0] + b[1] * x[2] * math.exp(-cg * x[3]) + b[2] * x[0] * x[1]
                  + b[3] * x[1] * x[1] + b[4] * x[0])
            res += (pr - yy) ** 2
        if best is None or res < best[0]:
            best = (res, cg, b)
    _, cg, b = best
    print("%r + %r * x3 * exp(%r * x4) + %r * x1 * x2 + %r * x2 * x2 + %r * x1"
          % (b[0], b[1], -cg, b[2], b[3], b[4]))


if __name__ == "__main__":
    main()
