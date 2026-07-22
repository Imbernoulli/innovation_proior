# TIER: greedy
"""The obvious first move: treat this as ordinary smooth regression on the
raw logged hold times, ignoring that some of them are capped. Fit ONE
quadratic  hold ~ b0 + b1*rho + b2*rho^2  by least squares over ALL training
rows (a flexible-looking basis, but still a single smooth polynomial with no
notion of a hang-up ceiling or a divergent pole). Because a meaningful chunk
of the high-load training rows are really "T or more" but got logged as
exactly T, the fitted curve is dragged down right where it matters most, and
a polynomial can never reproduce a pole anyway -- extrapolated to the
near-critical held-out loads it predicts a bounded, much-too-small hold
time."""
import sys


def solve_normal_eq(A, b):
    n = len(A[0])
    ATA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    ATb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(n)]
    M = [row[:] + [ATb[i]] for i, row in enumerate(ATA)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [M[r][k] - f * M[col][k] for k in range(n + 1)]
    return [M[i][n] for i in range(n)]


def main():
    data = sys.stdin.read().split()
    n_train = int(data[0])
    idx = 3
    rows = []
    for _ in range(n_train):
        rho = float(data[idx]); hold = float(data[idx + 1])
        idx += 2
        rows.append((rho, hold))

    A = [[1.0, rho, rho * rho] for rho, hold in rows]
    b = [hold for rho, hold in rows]
    b0, b1, b2 = solve_normal_eq(A, b)

    expr = "( %.8f ) + ( %.8f ) * rho + ( %.8f ) * rho ** 2" % (b0, b1, b2)
    print(expr)


if __name__ == "__main__":
    main()
