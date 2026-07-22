# TIER: greedy
# The obvious recipe: treat this as generic linear regression on the columns
# that are literally named in the objective -- the defect count D, the
# entropy H, and the motif count K -- and ignore the "bookkeeping" columns
# n, g, M, A entirely. Fit B ~ 1 + D + H + K by ordinary least squares.
#
# This fits the training ledger fine. But every training row has fold order
# g=8, so nothing in training can tell a linear model that the defect term
# is really a RATIO "defect per orbit" -- a straight line in raw D just
# calibrates a coefficient to the small D range the training grids happen to
# produce. On the held-out ledger the grids are bigger and the fold order
# varies (up to g=20), so raw D and raw K drift into a range the training
# coefficients were never calibrated for, and the linear extrapolation
# misses.
import sys


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-18:
            d = 1e-18
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-18 else 1e-18) for i in range(n)]


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    feats = []  # [1, D, H, K]
    y = []
    for i in range(n):
        D = float(vals[8 * i + 2])
        K = float(vals[8 * i + 5])
        H = float(vals[8 * i + 6])
        B = float(vals[8 * i + 7])
        feats.append([1.0, D, H, K])
        y.append(B)
    m = 4
    Amat = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, yy in zip(feats, y):
        for r in range(m):
            b[r] += x[r] * yy
            for c in range(m):
                Amat[r][c] += x[r] * x[c]
    c0, c1, c2, c3 = solve(Amat, b)
    print("%.10g + %.10g*D + %.10g*H + %.10g*K" % (c0, c1, c2, c3))


if __name__ == "__main__":
    main()
