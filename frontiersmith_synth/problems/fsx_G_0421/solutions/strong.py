# TIER: strong
# Chinchilla-style power law with an irreducible floor:
#     L(C, D) = E + A*C**(-alpha) + B*D**(-beta)
# Grid-search the two exponents (alpha, beta); for each pair the model is LINEAR
# in (E, A, B), solved by least squares.  Keep the exponent pair with the best
# training fit and emit the fitted closed form.  Because the floor is modelled,
# this extrapolates to the large-scale held-out split -- but the hidden
# exponents (only grid-approximated) plus the irreducible noise floor keep it
# short of a perfect score.
import sys, math


def solve3(M, y):
    A = [row[:] + [y[i]] for i, row in enumerate(M)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        if abs(A[c][c]) < 1e-12:
            A[c][c] = 1e-12
        pv = A[c][c]
        for j in range(c, 4):
            A[c][j] /= pv
        for r in range(3):
            if r != c:
                f = A[r][c]
                for j in range(c, 4):
                    A[r][j] -= f * A[c][j]
    return [A[0][3], A[1][3], A[2][3]]


def fit(rows, alpha, beta):
    M = [[0.0] * 3 for _ in range(3)]
    y = [0.0] * 3
    for C, D, L in rows:
        f = [1.0, C ** (-alpha), D ** (-beta)]
        for i in range(3):
            for j in range(3):
                M[i][j] += f[i] * f[j]
            y[i] += f[i] * L
    E, A, B = solve3(M, y)
    sse = 0.0
    for C, D, L in rows:
        pred = E + A * C ** (-alpha) + B * D ** (-beta)
        sse += (pred - L) ** 2
    return sse, E, A, B


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    rows = []
    for line in data[1:1 + n]:
        p = line.split()
        if len(p) < 3:
            continue
        rows.append((float(p[0]), float(p[1]), float(p[2])))

    best = None
    a = 0.15
    while a <= 0.451:
        b = 0.15
        while b <= 0.451:
            sse, E, A, B = fit(rows, a, b)
            if best is None or sse < best[0]:
                best = (sse, a, b, E, A, B)
            b += 0.01
        a += 0.01
    _, a, b, E, A, B = best
    sys.stdout.write("%r + %r*C**(%r) + %r*D**(%r)\n"
                     % (E, A, -a, B, -b))


if __name__ == "__main__":
    main()
