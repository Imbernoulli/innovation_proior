# TIER: strong
# Structured symbolic recovery of the period-luminosity relation.  Guess the
# physically-motivated feature family
#     [1, x1, x1^2, x2, x3, x3*x1]
# (log-period + curvature + color + metallicity zero-point + metallicity-
# dependent slope) and solve the linear coefficients by least squares.  This
# recovers the SHAPE of the hidden law, so it extrapolates onto the long-period
# tail far better than the linear fit; only the irreducible photometric-noise
# floor keeps it below 1.0.
import sys


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
    A = []; y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 4:
            v = list(map(float, p[:4]))
            A.append([1.0, v[0], v[0] * v[0], v[1], v[2], v[2] * v[0]])
            y.append(v[3])
    b = lstsq(A, y)
    print("%r + %r * x1 + %r * x1**2 + %r * x2 + %r * x3 + %r * x3 * x1"
          % (b[0], b[1], b[2], b[3], b[4], b[5]))


if __name__ == "__main__":
    main()
