# TIER: strong
# Recurrence discovery: fit an order-3 linear recurrence
#     a(n) = c1*a(n-1) + c2*a(n-2) + c3*a(n-3)
# to the observed prefix by least squares (values scaled for conditioning,
# a tiny ridge for stability), and emit it as a recurrence over a1,a2,a3.
# Recovers the dominant linear-recurrence signal of Fibonacci / n-nacci and
# low-degree figurate sequences, extrapolating far better than a polynomial;
# but it cannot represent order>=4 laws exactly and inherits the fresh
# far-future jitter through the supplied history, leaving headroom below 1.0.
import sys


def solve(M):
    n = len(M)
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
    T = int(data[0].split()[0])
    a = []
    for ln in data[1:1 + T]:
        p = ln.split()
        if len(p) >= 2:
            a.append(float(p[1]))
    scale = max(1.0, max(abs(v) for v in a))
    a = [v / scale for v in a]           # coefficients invariant under scaling
    # rows: target a[k], features [a[k-1], a[k-2], a[k-3]] for k in [3, T)
    A = [[0.0] * 4 for _ in range(3)]
    for k in range(3, T):
        feat = [a[k - 1], a[k - 2], a[k - 3]]
        for i in range(3):
            A[i][3] += feat[i] * a[k]
            for j in range(3):
                A[i][j] += feat[i] * feat[j]
    for i in range(3):
        A[i][i] += 1e-9                  # ridge
    c = solve(A)
    print("%r * a1 + %r * a2 + %r * a3" % (c[0], c[1], c[2]))


if __name__ == "__main__":
    main()
