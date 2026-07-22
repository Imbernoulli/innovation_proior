# TIER: strong
"""The insight: cST counts ADJACENT same-unit MUL pairs, but the multiply
unit is a single non-pipelined server.  A run of R consecutive multiplies
is a single-server queue whose service time exceeds its arrival interval,
so the backlog it leaves behind grows like a TRIANGULAR number of the run
length, not like a per-pair constant.  Since the probes are built from at
most one contiguous MUL cluster, the run length is exactly cST+1, so the
correct regressor is the triangular number

    u = cST*(cST+1)/2

Swap that single feature in for the raw cST count and refit the SAME linear
regression -- everything else about the recipe is identical to `greedy`,
only the feature is mechanistically motivated instead of curve-fit.  This
is what lets the fitted law transfer to the held-out probes, whose MUL
clusters run far longer than anything in the training table.
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
        u = cST * (cST + 1) / 2.0
        rows.append((n, cLU, cMF, u, cyc))

    X = [[1.0, r[0], r[1], r[2], r[3]] for r in rows]
    y = [r[4] for r in rows]
    b0, b1, b2, b3, b4 = solve_ridge(X, y)

    expr = ("%.6f + %.6f*n + %.6f*cLU + %.6f*cMF + %.6f*(cST*(cST+1)/2)"
             % (b0, b1, b2, b3, b4))
    print(expr)


if __name__ == "__main__":
    main()
