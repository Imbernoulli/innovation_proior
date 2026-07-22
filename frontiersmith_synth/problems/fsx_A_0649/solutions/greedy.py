# TIER: greedy
"""The obvious first move: treat this as ordinary smooth regression. Fit ONE
continuous model y ~ b0 + b1*F + b2*ln(1+r) + b3*F*ln(1+r) by least squares
over the whole training set (a flexible-looking basis, but still a single
formula with no branching). This ignores that the mechanism is a hybrid
automaton, so it averages the small "stuck" slope and the large "sliding"
slope into one intermediate slope -- systematically wrong near the guard and
under held-out extrapolation."""
import sys, math


def solve_normal_eq(A, b):
    """Solve (A^T A) x = A^T b for a small dense system via Gaussian
    elimination with partial pivoting. A is a list of rows, b a list."""
    n = len(A[0])
    ATA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    ATb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(n)]
    # augmented Gaussian elimination
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
    idx = 2
    rows = []
    for _ in range(n_train):
        F = float(data[idx]); r = float(data[idx + 1]); y = float(data[idx + 2])
        idx += 3
        L = math.log(1.0 + r)
        rows.append((F, L, y))

    A = [[1.0, F, L, F * L] for F, L, y in rows]
    b = [y for F, L, y in rows]
    coef = solve_normal_eq(A, b)
    b0, b1, b2, b3 = coef

    expr = ("( %.8f ) + ( %.8f ) * F + ( %.8f ) * log ( 1 + r ) "
            "+ ( %.8f ) * F * log ( 1 + r )") % (b0, b1, b2, b3)
    print(expr)


if __name__ == "__main__":
    main()
