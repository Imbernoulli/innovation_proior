# TIER: strong
"""Insight: on the visible sub-critical rows the true law
    q = (QM/RC)*rho - S0*P/(|rho-RC|+EPS0)
is a NONLINEAR regression in general (RC unknown), but for any FIXED
candidate RC it is LINEAR in (a,b) via the two features
    f1 = rho ,  f2 = P/(|rho-RC|+EPS0)
via  q ~= a*f1 + b*f2 .
That collapses the hard fit into a cheap 1-D search over candidate RC
(variable projection / profile least squares): for each candidate, solve
the 2-parameter linear least squares in closed form and score it by
residual SSE on the training rows; keep the RC that minimizes that SSE.
Only the GROWTH of the perturbation response as rho approaches the true
critical density identifies RC -- exactly the perturbation-susceptibility
mechanism this family is built around. Because the visible curvature is
mild, the profile-SSE surface is quite flat, so the raw arg-min is shrunk
30% toward a mild structural prior (RC is typically a modest multiple of
the largest observed sub-critical density) before a small downward safety
margin is applied so the reconstructed critical density never overshoots
into the held-out range (which starts only a couple of percent past it).

Once RC (hence capacity QM = a*RC and susceptibility scale S0 = -b) are
recovered, the held-out density range is known to sit ENTIRELY at or past
RC, so the metastable/discharge BLEND can be written as a single algebraic
expression with no branching needed:
    x      = (rho-RC)/(RJ-RC)                       in [0,1) on held-out
    q_pred = (1-x) * ( QM*(1 + K1*x) - S0*P/((rho-RC)+EPS0) )
  where K1 = (1-DELTA) - MU folds the metastable decline MU and the
  capacity-drop fraction DELTA into one constant, and the perturbation
  term keeps the SAME susceptibility law identified on the sub-critical
  rows (the fragility that produced the visible P-response there is the
  same fragility that governs the breakdown here).

RJ (jam density), MU and DELTA are never observable from sub-critical-only
training data -- they only manifest in the broken-down region this solver
never sees rows from. Deliberate simplification kept for a genuine, honest
generalization gap: RJ is guessed from RC via the family's typical
RC/RJ ratio, and MU/DELTA are guessed at their typical family midpoints
rather than fit per road (they vary a little per road), so this reference
stays short of the score ceiling."""
import sys

EPS0 = 3.0
FRAC_CRIT_MID = 0.33   # typical RC/RJ ratio across the family
MU_GUESS = 0.25
DELTA_GUESS = 0.425


def sse_for_RC(rows, RC):
    S11 = S12 = S22 = T1 = T2 = 0.0
    for rho, P, q in rows:
        denom = abs(rho - RC) + EPS0
        f1 = rho
        f2 = P / denom
        S11 += f1 * f1
        S12 += f1 * f2
        S22 += f2 * f2
        T1 += f1 * q
        T2 += f2 * q
    det = S11 * S22 - S12 * S12
    if abs(det) > 1e-9:
        a = (T1 * S22 - T2 * S12) / det
        b = (S11 * T2 - S12 * T1) / det
    else:
        a = T1 / S11 if S11 > 1e-9 else 0.0
        b = 0.0
    se = 0.0
    for rho, P, q in rows:
        denom = abs(rho - RC) + EPS0
        pred = a * rho + b * (P / denom)
        se += (q - pred) ** 2
    return se, a, b


def search_critical_density(rows):
    max_rho = max(rho for rho, P, q in rows)
    lo, hi = max_rho * 1.01, max_rho * 3.2
    best = None
    for _pass in range(6):
        n_grid = 40
        for i in range(n_grid):
            RC = lo + (hi - lo) * i / (n_grid - 1)
            se, a, b = sse_for_RC(rows, RC)
            if best is None or se < best[0]:
                best = (se, RC, a, b)
        step = (hi - lo) / n_grid
        lo = max(max_rho * 1.001, best[1] - 2 * step)
        hi = best[1] + 2 * step
    return best  # (se, RC, a, b)


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = []
    idx = 2
    for _ in range(n):
        rho = float(data[idx]); P = float(data[idx + 1]); q = float(data[idx + 2])
        idx += 3
        rows.append((rho, P, q))

    se, RC_est, a_est, b_est = search_critical_density(rows)

    # The profile-SSE surface over candidate RC is genuinely very flat (the
    # visible curvature is mild), so the raw arg-min is noisy, especially on
    # the harder/shorter training samples. Shrink it toward a mild structural
    # prior (critical density is typically a modest, family-wide multiple of
    # the largest observed sub-critical density) -- a standard bias/variance
    # trade-off that measurably lowers RC estimation error without needing
    # any additional information.
    max_rho = max(rho for rho, P, q in rows)
    RC_prior = max_rho * 1.35
    RC_est = 0.7 * RC_est + 0.3 * RC_prior

    RC_est *= 0.97  # safety margin: never overshoot into the held-out range
    QM_est = max(a_est * RC_est, 1.0)
    S0_est = max(-b_est, 0.0)

    RJ_est = RC_est / FRAC_CRIT_MID
    K1 = (1.0 - DELTA_GUESS) - MU_GUESS

    expr = (
        "(1 - (rho - %.6f) / (%.6f - %.6f)) * "
        "(%.6f * (1 + %.6f * (rho - %.6f) / (%.6f - %.6f)) "
        "- %.6f * P / ((rho - %.6f) + %.6f))"
        % (RC_est, RJ_est, RC_est,
           QM_est, K1, RC_est, RJ_est, RC_est,
           S0_est, RC_est, EPS0)
    )
    print(expr)


if __name__ == "__main__":
    main()
