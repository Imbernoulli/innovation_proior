# TIER: greedy
"""
The "average strong coder" move: a textbook log-linear (Arrhenius / power-law)
least-squares regression of log(R) on every column the input hands you --
log(Cl), 1/(T+273.15), log(tex+1), (pH-7)^2, and D thrown in as just another
linear feature, because why not use all the data you were given.

This fits the visible (all-passive) training rows beautifully. But R never
actually varies with D while the film is intact (excess-over-threshold is
exactly 0 for every training row), so D carries zero linear signal in-sample
and its fitted coefficient lands near 0. The resulting curve stays perfectly
smooth in Cl, T, pH, tex -- and is blind to the exponential breakdown once the
held-out environment crosses the (never-observed) chloride threshold.
"""
import sys, math


def ols(X, y):
    m = len(X); k = len(X[0])
    XtX = [[sum(X[r][i] * X[r][j] for r in range(m)) for j in range(k)] for i in range(k)]
    Xty = [sum(X[r][i] * y[r] for r in range(m)) for i in range(k)]
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    n = k
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[piv] = A[piv], A[i]
        if abs(A[i][i]) < 1e-12:
            continue
        for r in range(n):
            if r != i:
                f = A[r][i] / A[i][i]
                for c in range(i, n + 1):
                    A[r][c] -= f * A[i][c]
    return [A[i][n] / A[i][i] if abs(A[i][i]) > 1e-12 else 0.0 for i in range(n)]


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = data[2:]
    X, y = [], []
    for i in range(n):
        Cl, T, pH, tex, D, R = (float(v) for v in vals[i * 6:(i + 1) * 6])
        X.append([1.0, math.log(Cl), 1.0 / (T + 273.15), math.log(tex + 1.0), (pH - 7.0) ** 2, D])
        y.append(math.log(R))
    b = ols(X, y)

    expr = (
        "exp ( %.8e + %.8e * log ( Cl ) + %.8e * ( 1.0 / ( T + 273.15 ) ) "
        "+ %.8e * log ( tex + 1.0 ) + %.8e * ( pH - 7.0 ) ** 2 + %.8e * D )"
        % (b[0], b[1], b[2], b[3], b[4], b[5])
    )
    print(expr)


if __name__ == "__main__":
    main()
