# TIER: strong
"""Leading-indicator-gated piecewise model.

Insight: the training log is, by construction, entirely pre-knee -- every
target cycle in it sits before that cell's own (unobserved) knee -- so no
amount of curve-fitting on the visible (cyc, y) pairs alone can ever recover
the collapse. But the resistance-growth rate x2 is a noisy but genuine
leading indicator of the SAME internal stress state that determines when the
knee happens: cells run hotter / deeper cause both faster resistance growth
AND an earlier knee. So instead of only fitting the (very real, very linear)
pre-knee trend from the data, this solution ALSO uses x2 to estimate each
row's own knee-onset cycle, and forecasts a SMOOTH gated transition from the
fitted linear pre-knee line to a multiplicative post-knee collapse anchored
at that estimated onset -- built entirely from the whitelisted operators
(a tanh soft-gate stands in for the "if cyc <= knee" branch).

The linear pre-knee slope/intercept ARE fit from the data (least squares on
(1, x4)); the resistance -> knee-onset mapping and the post-knee collapse
rate are physically-motivated priors about the SHAPE of the phenomenon
(stated in the problem) -- exact calibration of those is not recoverable
from a pre-knee-only log, so residual error remains (no saturation)."""
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
    y = [r[5] for r in rows]

    # fit the (genuinely recoverable) near-linear pre-knee trend
    A = [[1.0, r[4]] for r in rows]
    a0, a1 = lstsq(A, y)

    # leading-indicator-informed knee-onset estimate: nk_hat = N0G - KAPPAG*x2
    N0G = 910.0
    KAPPAG = 200.0
    GATE_K = 0.02
    BETAG = 0.006

    nk = "(%.6f - (%.6f)*x2)" % (N0G, KAPPAG)
    gate = "(0.5*(1.0+tanh(%.6f*(%s - x4))))" % (GATE_K, nk)
    y_pre = "(%.8f + (%.8f)*x4)" % (a0, a1)
    y_knee = "(%.8f + (%.8f)*%s)" % (a0, a1, nk)
    y_post = "((%s)*exp(-(%.6f)*(x4 - %s)))" % (y_knee, BETAG, nk)

    expr = "(%s)*(%s) + (%s)*(1.0 - %s)" % (y_pre, gate, y_post, gate)
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()
