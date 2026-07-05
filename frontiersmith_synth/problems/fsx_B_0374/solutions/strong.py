# TIER: strong
"""Floored scaling law: L = E_inf + A*C^-alpha + B*R^-beta.
Grid-search the two exponents; for each (alpha,beta) solve linear least squares
for (E_inf, A, B). Pick the fit with the lowest training residual. Because it
recovers the irreducible floor E_inf, it extrapolates well into the large-C,R
held-out region."""
import sys


def solve_lin(rows, ys, k):
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for r, y in zip(rows, ys):
        for a in range(k):
            atb[a] += r[a] * y
            for b in range(k):
                ata[a][b] += r[a] * r[b]
    M = [ata[i][:] + [atb[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c] if abs(M[c][c]) > 1e-12 else 1e-12
        for j in range(c, k + 1):
            M[c][j] /= pv
        for r in range(k):
            if r != c:
                f = M[r][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][k] for i in range(k)]


def main():
    data = sys.stdin.read().split("\n")
    hdr = data[0].split()
    n = int(hdr[0])
    C = []
    R = []
    Y = []
    for i in range(1, n + 1):
        c, r, l = (float(v) for v in data[i].split())
        C.append(c)
        R.append(r)
        Y.append(l)

    best = None
    # coarse then fine could be used; a single moderate grid suffices
    grid = [0.20 + 0.025 * k for k in range(0, 21)]  # 0.20 .. 0.70
    for alpha in grid:
        ca = [c ** (-alpha) for c in C]
        for beta in grid:
            rb = [r ** (-beta) for r in R]
            rows = [(1.0, ca[i], rb[i]) for i in range(n)]
            coef = solve_lin(rows, Y, 3)
            E, A, B = coef
            sse = 0.0
            for i in range(n):
                pred = E + A * ca[i] + B * rb[i]
                d = pred - Y[i]
                sse += d * d
            if best is None or sse < best[0]:
                best = (sse, E, A, B, alpha, beta)

    _, E, A, B, alpha, beta = best
    sys.stdout.write("%r + %r*x1**(%r) + %r*x2**(%r)\n"
                     % (E, A, -alpha, B, -beta))


if __name__ == "__main__":
    main()
