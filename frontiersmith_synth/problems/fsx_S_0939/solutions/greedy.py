# TIER: greedy
"""The obvious first move: an ADDITIVE per-hazard-count linear regression.

cycles ~ b0 + b1*n + b2*cLU + b3*cMF + b4*cST

fit by ordinary least squares on the training table.  This treats every
extra adjacent same-unit MUL pair as costing one more constant increment,
which is exactly what the short training clusters (0-2 adjacent pairs) look
like.  It says nothing about how a long run of consecutive multiplies
behaves, because the training data never shows one.
"""
import sys


def solve_ridge(X, y, ridge=1e-6):
    m = len(X[0])
    XtX = [[0.0] * m for _ in range(m)]
    Xty = [0.0] * m
    for row, yv in zip(X, y):
        for i in range(m):
            Xty[i] += row[i] * yv
            for j in range(m):
                XtX[i][j] += row[i] * row[j]
    for i in range(m):
        XtX[i][i] += ridge
    A = [XtX[i][:] + [Xty[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        if abs(pv) < 1e-12:
            pv = 1e-12
        for j in range(col, m + 1):
            A[col][j] /= pv
        for r in range(m):
            if r != col:
                f = A[r][col]
                if f != 0.0:
                    for j in range(col, m + 1):
                        A[r][j] -= f * A[col][j]
    return [A[i][m] for i in range(m)]


def main():
    data = sys.stdin.read().split()
    n_train = int(data[0])
    vals = data[2:]
    rows = []
    for i in range(n_train):
        chunk = vals[9 * i: 9 * i + 9]
        n, nA, nM, nL, nS, cLU, cMF, cST = (int(x) for x in chunk[:8])
        cyc = float(chunk[8])
        rows.append((n, cLU, cMF, cST, cyc))

    X = [[1.0, r[0], r[1], r[2], r[3]] for r in rows]
    y = [r[4] for r in rows]
    b0, b1, b2, b3, b4 = solve_ridge(X, y)

    expr = "%.6f + %.6f*n + %.6f*cLU + %.6f*cMF + %.6f*cST" % (b0, b1, b2, b3, b4)
    print(expr)


if __name__ == "__main__":
    main()
