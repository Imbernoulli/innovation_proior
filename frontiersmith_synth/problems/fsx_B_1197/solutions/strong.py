# TIER: strong
# The insight: do NOT assume the repeat period is exactly one day.  A
# memoryless "fit harmonics of exactly 86400s" (the greedy recipe) explains
# the visible training days almost as well as anything else COULD, because
# the true orbital-repeat period P1 is only a few hundred seconds off -- but
# that residual gap is exactly what accumulates into a large phase error over
# the held-out horizon weeks later.
#
# So: jointly fit (a) the KNOWN-exact solar term at 86400s and (b) a
# 1st/2nd/3rd-harmonic series at a CANDIDATE period P1, grid-searching P1
# (coarse, then refined).  Critically, only TRUST the searched period if it
# explains meaningfully more training variance per extra parameter than the
# safe pure-24h model (an F-test-style check against a hand-picked, look-
# elsewhere-inflated threshold) -- otherwise fall back to the pure-24h fit.
# This is what keeps the insight from turning into "chase whatever period
# best fits the noise": on genuinely resolvable instances it locks onto the
# true near-day period and stays in phase far into the held-out horizon; on
# instances where the training window is too short/noisy to say anything
# with confidence, it safely reproduces the greedy recipe instead of
# overfitting to a spurious period.
import sys, math
import numpy as np

SOLAR_PERIOD = 86400.0
F_THRESH = 5.0


def design_fixed(t, w0):
    return np.stack([
        np.ones_like(t), np.cos(w0 * t), np.sin(w0 * t),
        np.cos(2 * w0 * t), np.sin(2 * w0 * t),
        np.cos(3 * w0 * t), np.sin(3 * w0 * t),
    ], axis=1)


def design_joint(t, w0, w1):
    return np.stack([
        np.ones_like(t), np.cos(w0 * t), np.sin(w0 * t),
        np.cos(w1 * t), np.sin(w1 * t),
        np.cos(2 * w1 * t), np.sin(2 * w1 * t),
        np.cos(3 * w1 * t), np.sin(3 * w1 * t),
    ], axis=1)


def sse_joint(t, y, w0, gap):
    P1 = SOLAR_PERIOD - gap
    w1 = 2 * math.pi / P1
    A = design_joint(t, w0, w1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    sse = float(np.sum((A @ coef - y) ** 2))
    return sse, coef


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0")
        return
    n = int(data[0])
    vals = data[2:]
    t = np.array([float(vals[2 * i]) for i in range(n)], dtype=np.float64)
    y = np.array([float(vals[2 * i + 1]) for i in range(n)], dtype=np.float64)

    w0 = 2 * math.pi / SOLAR_PERIOD

    # coarse grid over candidate repeat-period "gap below 24h" (seconds)
    best = None
    best_gap = None
    best_coef = None
    for gap in range(80, 461, 4):
        sse, coef = sse_joint(t, y, w0, gap)
        if best is None or sse < best:
            best, best_gap, best_coef = sse, gap, coef

    # refine around the coarse winner
    lo = max(20.0, best_gap - 5.0)
    hi = min(700.0, best_gap + 5.0)
    step = 0.05
    ngrid = int(round((hi - lo) / step)) + 1
    for k in range(ngrid):
        gap = lo + k * step
        sse, coef = sse_joint(t, y, w0, gap)
        if sse < best:
            best, best_gap, best_coef = sse, gap, coef

    # pure-24h reference fit (7 params) + F-test-style acceptance gate
    A0 = design_fixed(t, w0)
    coef0, *_ = np.linalg.lstsq(A0, y, rcond=None)
    e24 = float(np.sum((A0 @ coef0 - y) ** 2))
    dof_full = max(1, n - 9)
    Fstat = ((e24 - best) / 2.0) / max(1e-12, best / dof_full)

    if Fstat > F_THRESH:
        P1 = SOLAR_PERIOD - best_gap
        w1 = 2 * math.pi / P1
        c0, a0, b0, a1, b1, a2, b2, a3, b3 = [float(x) for x in best_coef]
        expr = (
            "%.8f + %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t ) "
            "+ %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t ) "
            "+ %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t ) "
            "+ %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t )"
            % (c0,
               a0, w0, b0, w0,
               a1, w1, b1, w1,
               a2, 2 * w1, b2, 2 * w1,
               a3, 3 * w1, b3, 3 * w1)
        )
    else:
        c0, a0, b0, a1, b1, a2, b2 = [float(x) for x in coef0]
        expr = (
            "%.8f + %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t ) "
            "+ %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t ) "
            "+ %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t )"
            % (c0,
               a0, w0, b0, w0,
               a1, 2 * w0, b1, 2 * w0,
               a2, 3 * w0, b2, 3 * w0)
        )
    print(expr)


if __name__ == "__main__":
    main()
