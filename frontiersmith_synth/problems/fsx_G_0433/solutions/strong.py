# TIER: strong
# Focus-structured recovery (Kepler's first law).  Because the Sun sits at a
# FOCUS (the origin), the orbit obeys  r = p - e*(cos w * x + sin w * y)  with
# r = sqrt(x^2+y^2), i.e. the special 3-parameter conic
#     A*r + B*x + C*y = 1,   A = 1/p, B = e*cos w /p, C = e*sin w /p.
# Fit A,B,C by linear least squares on rows [r, x, y] -> 1.  This is the exact
# hidden structure with only three parameters, so it extrapolates onto the
# withheld arc far better than the generic quadratic and pays a smaller
# complexity penalty; only the irreducible positional-noise floor keeps it
# below 1.0.
import sys
import math


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
            r = math.hypot(x, y)
            A.append([r, x, y])
            b.append(1.0)
    c = lstsq(A, b)
    print("%r * sqrt(x**2 + y**2) + %r * x + %r * y - 1" % (c[0], c[1], c[2]))


if __name__ == "__main__":
    main()
