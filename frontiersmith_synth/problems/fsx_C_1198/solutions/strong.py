# TIER: strong
"""Insight: the true law (alpha + beta*B^2) * L / (C - L) is a NONLINEAR
3-parameter regression in general, but for any FIXED capacity C it is
LINEAR in (alpha, beta) via the features L/(C-L) and B^2*L/(C-L). That
collapses the hard 3-D nonlinear fit into a cheap 1-D search over candidate
C (variable projection / profile least squares): for each candidate C,
solve the 2-parameter linear least squares in closed form and score it by
residual SSE on the (mildly curved) training rows; keep the C that
minimizes that SSE.

Only the ratio of curvature to slope in the visible sub-saturation range
identifies C -- exactly the mechanism the family is built around. A small
safety margin is applied so the reconstructed capacity never undershoots
(which would flip the sign of the denominator on held-out loads closer to
the true capacity).

Deliberate simplification kept for a genuine, honest generalization gap:
this solver hard-codes the burstiness exponent at 2 (the textbook Kingman
value) rather than also profiling over it -- the true exponent drifts a
little per system, so this reference stays short of the score ceiling."""
import sys


def sse_for_C(rows, C):
    S11 = S12 = S22 = T1 = T2 = 0.0
    for L, B, W in rows:
        denom = C - L
        if denom <= 1e-6:
            return None
        f1 = L / denom
        f2 = (B * B * L) / denom
        S11 += f1 * f1
        S12 += f1 * f2
        S22 += f2 * f2
        T1 += f1 * W
        T2 += f2 * W
    det = S11 * S22 - S12 * S12
    if abs(det) > 1e-12:
        a = (T1 * S22 - T2 * S12) / det
        b = (S11 * T2 - S12 * T1) / det
    else:
        a = T1 / S11 if S11 > 1e-12 else 0.0
        b = 0.0
    se = 0.0
    for L, B, W in rows:
        denom = C - L
        pred = a * (L / denom) + b * ((B * B * L) / denom)
        se += (W - pred) ** 2
    return se, a, b


def search_capacity(rows):
    max_L = max(L for L, B, W in rows)
    lo, hi = max_L * 1.01, max_L * 60.0
    best = None
    for _pass in range(5):
        n_grid = 40
        for i in range(n_grid):
            C = lo + (hi - lo) * i / (n_grid - 1)
            res = sse_for_C(rows, C)
            if res is None:
                continue
            se, a, b = res
            if best is None or se < best[0]:
                best = (se, C, a, b)
        step = (hi - lo) / n_grid
        lo = max(max_L * 1.001, best[1] - 2 * step)
        hi = best[1] + 2 * step
    return best  # (se, C, a, b)


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = []
    idx = 2
    for _ in range(n):
        L = float(data[idx]); B = float(data[idx + 1]); W = float(data[idx + 2])
        idx += 3
        rows.append((L, B, W))

    se, C_est, alpha_est, beta_est = search_capacity(rows)
    C_est *= 1.04  # safety margin against undershoot near held-out loads
    alpha_est = max(alpha_est, 1e-6)
    beta_est = max(beta_est, 0.0)

    print("( %.6f + %.6f * B ** 2 ) * L / ( %.6f - L )" % (alpha_est, beta_est, C_est))


if __name__ == "__main__":
    main()
