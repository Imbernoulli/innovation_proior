# TIER: greedy
# Generic algebraic conic fit.  The orbit does not pass through the Sun (origin
# is an interior focus), so the constant term is nonzero and we can normalise it
# to -1 and solve an ordinary linear least-squares problem for the general conic
#     A*x^2 + B*x*y + C*y^2 + D*x + E*y = 1  ->  F = A*x^2+B*x*y+C*y^2+D*x+E*y-1.
# This captures the quadratic shape but the algebraic (unweighted) fit is biased
# and, being a full 6-term quadratic, pays a larger complexity penalty than the
# structured focus form, so it extrapolates onto the withheld arc less cleanly.
import sys


def lstsq(A, y):
    m = len(A)
    n = len(A[0])
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
    A = []
    b = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 2:
            x = float(p[0])
            y = float(p[1])
            A.append([x * x, x * y, y * y, x, y])
            b.append(1.0)
    c = lstsq(A, b)
    print("%r * x**2 + %r * x*y + %r * y**2 + %r * x + %r * y - 1"
          % (c[0], c[1], c[2], c[3], c[4]))


if __name__ == "__main__":
    main()
