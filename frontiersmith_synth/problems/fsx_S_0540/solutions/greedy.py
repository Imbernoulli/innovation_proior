# TIER: greedy
# The obvious recipe: fit a flexible black-box polynomial of the pre-state to
# the observed v1 by ordinary least squares -- v1 ~ [1, u1, u2, u1*u2, u1^2,
# u2^2].  It interpolates the gentle low-energy logbook almost perfectly, but it
# models the MAP directly and never touches the conserved quantities, so its
# polynomial tail over-shoots the velocity ceiling and it extrapolates badly on
# the violent high-energy held-out shots.
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
    y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 4:
            u1, u2, v1 = float(p[0]), float(p[1]), float(p[2])
            A.append([1.0, u1, u2, u1 * u2, u1 * u1, u2 * u2])
            y.append(v1)
    b = lstsq(A, y)
    print("%r + %r*u1 + %r*u2 + %r*u1*u2 + %r*u1*u1 + %r*u2*u2"
          % (b[0], b[1], b[2], b[3], b[4], b[5]))


if __name__ == "__main__":
    main()
