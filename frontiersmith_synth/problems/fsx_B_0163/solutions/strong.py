# TIER: strong
"""Structure-recovering least squares. Searches a small library of candidate
poly-exp bases (a few exponential rates), selects the basis with the best training
fit, and emits the fitted closed form. It recovers the true functional shape
{1, x0^2, x1*x2, exp(k*x3), x1}, so it extrapolates well into the held-out storm
regime -- but training noise leaves residual coefficient error, so it does not
saturate."""
import sys, math


def lstsq(A, b):
    n = len(A[0])
    ATA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    ATb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(n)]
    M = [ATA[i][:] + [ATb[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c] or 1e-12
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def fit_for_k(rows, y, k):
    A = [[1.0, r[0]**2, r[1]*r[2], math.exp(k*r[3]), r[1]] for r in rows]
    c = lstsq(A, y)
    sse = 0.0
    for r, yy in zip(rows, y):
        pred = c[0] + c[1]*r[0]**2 + c[2]*r[1]*r[2] + c[3]*math.exp(k*r[3]) + c[4]*r[1]
        sse += (pred - yy) ** 2
    return c, sse


def main():
    vals = [float(t) for t in sys.stdin.read().split()]
    rows = [vals[i:i + 5] for i in range(0, len(vals), 5)]
    y = [r[4] for r in rows]
    best = None
    for kk in [0.30, 0.40, 0.45, 0.50, 0.60]:
        c, sse = fit_for_k(rows, y, kk)
        if best is None or sse < best[0]:
            best = (sse, kk, c)
    _, k, c = best
    expr = ("%.6f + (%.6f)*x0**2 + (%.6f)*x1*x2 + (%.6f)*exp(%.4f*x3) + (%.6f)*x1"
            % (c[0], c[1], c[2], c[3], k, c[4]))
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()
