# TIER: strong
"""
The insight: D is not "just another regression feature" -- it is a
REPASSIVATION-MARGIN diagnostic. Every training coupon has D > 0 (pulled
before its film broke down), so D is genuinely uninformative BY CORRELATION
inside the training data; a data-only fit correctly learns to ignore it (see
greedy.py). But the mechanism note says the margin's SIGN, not its
in-sample correlation, is what predicts the coming regime change: once an
untested chemistry pushes D negative, the film has locally failed and the
rate should jump by orders of magnitude.

So: fit the SAME smooth passive-regime trend as greedy (excluding D, since
excess is identically 0 for every training row and D itself is fully
determined by Cl,T,pH given the unknown threshold -- nothing here recovers
the exact per-instance jump-rate or curvature), then multiply in a universal
exponential breakdown factor keyed on max(0,-D). This closes most of the
orders-of-magnitude gap on the held-out aggressive environments without ever
having seen an active-regime training example. The exact jump-rate and
curvature are never revealed by the (all-passive) log, so this reference
deliberately does NOT chase a perfect per-instance fit -- headroom remains
for a solver that infers or bounds the curvature more cleverly.
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
        X.append([1.0, math.log(Cl), 1.0 / (T + 273.15), math.log(tex + 1.0), (pH - 7.0) ** 2])
        y.append(math.log(R))
    b = ols(X, y)

    K = 7.0       # universal guessed breakdown exponential rate (mechanism note only)
    gamma = 0.9   # universal guessed breakdown linear pre-factor

    expr = (
        "exp ( %.8e + %.8e * log ( Cl ) + %.8e * ( 1.0 / ( T + 273.15 ) ) "
        "+ %.8e * log ( tex + 1.0 ) + %.8e * ( pH - 7.0 ) ** 2 ) "
        "* ( 1.0 + %.8e * max ( 0.0 , -D ) ) * exp ( %.8e * max ( 0.0 , -D ) )"
        % (b[0], b[1], b[2], b[3], b[4], gamma, K)
    )
    print(expr)


if __name__ == "__main__":
    main()
