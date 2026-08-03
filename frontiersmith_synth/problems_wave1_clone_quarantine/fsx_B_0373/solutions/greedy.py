# TIER: greedy
# Ordinary least-squares LINEAR fit y ~ 1 + x1 + x2 + x3 + x4.
# Captures the first-order trend but misses the demand*staleness exponential
# envelope, the flour*proof-temp interaction and the proof-temp square-law, so
# it extrapolates poorly onto the over-range frontier.
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
        if len(p) >= 5:
            v = list(map(float, p[:5]))
            A.append([1.0, v[0], v[1], v[2], v[3]])
            y.append(v[4])
    b = lstsq(A, y)
    print("%r + %r * x1 + %r * x2 + %r * x3 + %r * x4"
          % (b[0], b[1], b[2], b[3], b[4]))


if __name__ == "__main__":
    main()
