# TIER: greedy
"""Quadratic-diagonal least squares: basis {1, x0, x1, x2, x3, x0^2, x1^2, x2^2,
x3^2}. Captures per-axis curvature (so it beats the constant baseline out of
distribution) but has no cross-product (x1*x2) or exponential term, so it
under-extrapolates the true poly-exp law -> a middle score."""
import sys


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


def main():
    vals = [float(t) for t in sys.stdin.read().split()]
    rows = [vals[i:i + 5] for i in range(0, len(vals), 5)]
    A = [[1.0, r[0], r[1], r[2], r[3], r[0]**2, r[1]**2, r[2]**2, r[3]**2] for r in rows]
    y = [r[4] for r in rows]
    c = lstsq(A, y)
    terms = ["%.6f" % c[0]]
    names = ["x0", "x1", "x2", "x3", "x0**2", "x1**2", "x2**2", "x3**2"]
    for coef, nm in zip(c[1:], names):
        terms.append("(%.6f)*%s" % (coef, nm))
    sys.stdout.write(" + ".join(terms) + "\n")


if __name__ == "__main__":
    main()
