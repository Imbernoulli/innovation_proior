# TIER: greedy
"""Full linear least-squares fit: y ~ a0 + a1*x0 + a2*x1 + a3*x2 + a4*x3 + a5*x4.
This is the obvious first move -- throw every column (including the resistance
signal x2) into one ordinary linear regression and extrapolate the target cycle
x4 forward. It fits the near-linear pre-knee training log well (all training
targets ARE pre-knee, so a straight line is nearly exact there), and it even
picks up a mild positive/negative tilt from x2 -- but a straight line cannot
represent a sharp multiplicative collapse, so once the held-out horizon carries
a cell past its own (never-seen) knee cycle, this model just keeps gliding down
the same shallow line instead of falling off a cliff -> badly overshoots the
true remaining capacity on every post-knee case."""
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
    rows = [vals[i:i + 6] for i in range(0, len(vals), 6)]
    A = [[1.0, r[0], r[1], r[2], r[3], r[4]] for r in rows]
    y = [r[5] for r in rows]
    c = lstsq(A, y)
    names = ["x0", "x1", "x2", "x3", "x4"]
    terms = ["%.8f" % c[0]]
    for coef, nm in zip(c[1:], names):
        terms.append("(%.8f)*%s" % (coef, nm))
    sys.stdout.write(" + ".join(terms) + "\n")


if __name__ == "__main__":
    main()
