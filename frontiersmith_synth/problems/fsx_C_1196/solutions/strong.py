# TIER: strong
"""
Discrete-logistic invariant + hint shrinkage + regime-aware extrapolation.

Insight 1 (reformulation): the true recursion A(t+1) = A(t) + k*A(t)*(1-A(t)/M)
implies the per-capita growth rate g(t) = (A(t+1)-A(t))/A(t) is LINEAR in
A(t): g(t) = k - (k/M)*A(t). This is measurable from consecutive training
pairs even while A(t) is a small fraction of M -- you do not need to SEE
saturation to detect its rate, only to read the (faint) slope of per-capita
growth against level. OLS on that invariant recovers (k, M0) directly.

Insight 2 (shrinkage): the invariant's slope is a weak, noisy signal on
slow-growth / late-launch training windows (little curvature to see). Blend
the data-driven M0 estimate with the analyst M_hint, weighted by the
regression's own R^2 -- trust the data more when the data actually shows
curvature, trust the hint more when it does not.

Insight 3 (regime-aware extrapolation): the true ceiling steps DOWN at the
announced launch step t_B. Extrapolate the pre-launch logistic (same k, M0)
forward to t_B, apply the severity hint to compute the post-launch ceiling,
and emit a TWO-PIECE closed-form logistic joined at t_B -- carrying the
correct ceiling, not just the correct early rate, across the regime change.
"""
import sys, math


def ols(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        slope = 0.0
    else:
        slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def main():
    data = sys.stdin.read().split()
    idx = 0
    tid = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    tB = float(data[idx]); idx += 1
    s_hint = float(data[idx]); idx += 1
    M_hint = float(data[idx]); idx += 1
    rows = []
    for _ in range(n):
        ti = float(data[idx]); idx += 1
        A = float(data[idx]); idx += 1
        rows.append((ti, A))
    rows.sort(key=lambda r: r[0])
    N_train = rows[-1][0]
    A_last = rows[-1][1]

    # ---- Insight 1: per-capita growth invariant regression ----
    xs, ys = [], []
    for i in range(len(rows) - 1):
        t0, a0 = rows[i]
        t1, a1 = rows[i + 1]
        if t1 - t0 != 1:
            continue
        g = (a1 - a0) / a0
        xs.append(a0); ys.append(g)
    slope, intercept = ols(xs, ys)
    k_hat = intercept
    if slope < -1e-9 and k_hat > 1e-6:
        M0_data = -k_hat / slope
    else:
        M0_data = M_hint

    # R^2 of the invariant regression: how trustworthy is the data slope?
    ybar = sum(ys) / len(ys)
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 0.0 if ss_tot < 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)
    w = max(0.10, min(0.70, r2))

    # ---- Insight 2: shrink data-driven M0 toward the analyst hint ----
    M0_data = max(A_last * 1.05, min(M_hint * 3.0, M0_data))
    M0_hat = w * M0_data + (1.0 - w) * M_hint
    M0_hat = max(A_last * 1.05, M0_hat)
    k_hat = max(0.02, min(0.8, k_hat))

    # Pre-launch piece, anchored exactly at (N_train, A_last).
    C1 = (M0_hat - A_last) / A_last

    # ---- Insight 3: extrapolate to t_B, apply severity, join at t_B ----
    A_tB = M0_hat / (1.0 + C1 * math.exp(-k_hat * (tB - N_train)))
    M1_hat = A_tB + (1.0 - s_hint) * (M0_hat - A_tB)
    M1_hat = max(A_tB * 1.001, M1_hat)
    C2 = (M1_hat - A_tB) / A_tB

    pre = "%.10g / (1.0 + %.10g * expv(-%.10g * (t - %.10g)))" % (M0_hat, C1, k_hat, N_train)
    post = "%.10g / (1.0 + %.10g * expv(-%.10g * (t - %.10g)))" % (M1_hat, C2, k_hat, tB)

    print("(%s) if t < %.10g else (%s)" % (pre, tB, post))


if __name__ == "__main__":
    main()
