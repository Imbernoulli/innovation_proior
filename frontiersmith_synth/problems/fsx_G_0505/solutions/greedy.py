# TIER: greedy
import math
import sys


def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-10:
            M[c][c] = 1e-10
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r == c:
                continue
            f = M[r][c]
            if f:
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def basis(N, T, D, R):
    u = N / 100.0
    th = (T - 20.0) / 12.0
    return [
        1.0,
        math.sqrt(u),
        u,
        th,
        D,
        R,
        u * th,
        D * R,
        u * D * R,
    ]


def main():
    lines = sys.stdin.read().strip().splitlines()
    n = int(lines[0].split()[0])
    rows = []
    for line in lines[1:1 + n]:
        p = line.split()
        rows.append(tuple(float(x) for x in p[:5]))

    m = 9
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    ridge = 1e-5
    for N, T, D, R, y in rows:
        f = basis(N, T, D, R)
        for i in range(m):
            b[i] += f[i] * y
            for j in range(m):
                A[i][j] += f[i] * f[j]
    for i in range(m):
        A[i][i] += ridge
    c = solve(A, b)
    expr = (
        "%r + %r*sqrt(N/100.0) + %r*(N/100.0) + %r*((T-20.0)/12.0) "
        "+ %r*D + %r*R + %r*(N/100.0)*((T-20.0)/12.0) "
        "+ %r*D*R + %r*(N/100.0)*D*R"
    ) % tuple(c)
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()
