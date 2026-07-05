# TIER: strong
# Structured symbolic search for the vol surface: guess the feature family
#     [1, t, k, k^2, k^2 * exp(-r*t)]
# and grid-search the hidden smile-decay rate r, picking the best training fit,
# then solve the linear coefficients by least squares.  Recovers the wing
# convexity so it extrapolates into the far strikes far better than the linear
# fit, but the coarse r-grid + quote noise + irreducible held-out floor leave
# residual headroom below 1.0.
import sys, math


def lstsq(A, y):
    m = len(A); n = len(A[0])
    M = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(m):
        for j in range(n):
            M[j][n] += A[i][j] * y[i]
            for kk in range(n):
                M[j][kk] += A[i][j] * A[i][kk]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col] or 1e-12
        for kk in range(col, n + 1):
            M[col][kk] /= pv
        for r in range(n):
            if r != col:
                f = M[r][col]
                for kk in range(col, n + 1):
                    M[r][kk] -= f * M[col][kk]
    return [M[j][n] for j in range(n)]


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    X = []; y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 3:
            v = list(map(float, p[:3]))
            X.append((v[0], v[1])); y.append(v[2])
    best = None
    for rg in [0.8, 1.1, 1.4, 1.7]:
        A = [[1.0, t, k, k * k, k * k * math.exp(-rg * t)] for (k, t) in X]
        b = lstsq(A, y)
        res = 0.0
        for (k, t), yy in zip(X, y):
            pr = (b[0] + b[1] * t + b[2] * k + b[3] * k * k
                  + b[4] * k * k * math.exp(-rg * t))
            res += (pr - yy) ** 2
        if best is None or res < best[0]:
            best = (res, rg, b)
    _, rg, b = best
    print("%r + %r * t + %r * k + %r * k**2 + %r * k**2 * exp(%r * t)"
          % (b[0], b[1], b[2], b[3], b[4], -rg))


if __name__ == "__main__":
    main()
