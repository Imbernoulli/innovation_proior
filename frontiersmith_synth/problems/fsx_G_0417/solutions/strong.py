# TIER: strong
"""Structure-recovering fit.  Searches a small library of short-range correction
exponents {r^-3, r^-4, r^-5} paired with the leading r^-2 term, selects the basis
with the best training fit (which recovers the true r^-4 structural correction),
and emits the fitted closed form F = A/r^2 + C/r^k.  Because it recovers the true
functional shape it extrapolates well into the held-out outer band -- but training
noise leaves residual coefficient error, so it does not saturate."""
import sys


def lstsq(A, b):
    n = len(A[0])
    ATA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    ATb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(n)]
    M = [ATA[i][:] + [ATb[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda rr: abs(M[rr][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c] or 1e-12
        M[c] = [v / pv for v in M[c]]
        for rr in range(n):
            if rr != c:
                f = M[rr][c]
                M[rr] = [a - f * bb for a, bb in zip(M[rr], M[c])]
    return [M[i][n] for i in range(n)]


def fit_for_k(rows, k):
    Amat = [[1.0 / (r * r), 1.0 / (r ** k)] for (r, f) in rows]
    y = [f for (r, f) in rows]
    c = lstsq(Amat, y)
    sse = 0.0
    for (r, f) in rows:
        pred = c[0] / (r * r) + c[1] / (r ** k)
        sse += (pred - f) ** 2
    return c, sse


def main():
    vals = [float(t) for t in sys.stdin.read().split()]
    rows = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    best = None
    for k in (3, 4, 5):
        c, sse = fit_for_k(rows, k)
        if best is None or sse < best[0]:
            best = (sse, k, c)
    _, k, c = best
    sys.stdout.write("%.10f / r**2 + %.10f / r**%d\n" % (c[0], c[1], k))


if __name__ == "__main__":
    main()
