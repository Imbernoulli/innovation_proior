# TIER: greedy
# Ordinary least-squares LINEAR fit sigma ~ 1 + k + t.
# Captures the equity skew (linear in k) and term structure, but misses the
# k^2 smile convexity, so it extrapolates poorly into the far-strike wings.
import sys


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
    A = []; y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 3:
            v = list(map(float, p[:3]))
            A.append([1.0, v[0], v[1]])
            y.append(v[2])
    b = lstsq(A, y)
    print("%r + %r * k + %r * t" % (b[0], b[1], b[2]))


if __name__ == "__main__":
    main()
