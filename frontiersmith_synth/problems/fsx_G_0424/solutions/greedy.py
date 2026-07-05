# TIER: greedy
# Log-linear fit  Cd ~ p + q*log(Re).
# Captures the mid-Re downward trend, but a straight line in log(Re) has the
# wrong asymptote and over-drops at high Re, so it extrapolates only modestly.
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
    A = []; y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 3:
            v = list(map(float, p[:3]))
            A.append([1.0, math.log(v[0])])
            y.append(v[2])
    b = lstsq(A, y)
    print("%r + %r * log(Re)" % (b[0], b[1]))


if __name__ == "__main__":
    main()
