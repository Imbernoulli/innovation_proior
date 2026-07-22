# TIER: strong
"""
Insight: stop fitting a single curve to the whole band and instead SEARCH for
the regime boundary xc under which the data reformulates into a clean power
law.

For each candidate split point xc_try (scanned over the interior of the
training x-range):
  1. Fit a quadratic y = c0 + c1*x + c2*x^2 (ordinary least squares, solved
     via the 3x3 normal equations) to only the points LEFT of xc_try -- the
     calm regime is genuinely quadratic there.
  2. Anchor y0 = quadratic(xc_try) and take the RIGHT-of-xc_try residuals
     r_i = y_i - y0. Where r_i > 0, fit log(r_i) = log(B) + alpha*log(x_i -
     xc_try) by ordinary least squares in the transformed (log-log) variables
     -- this is the "right variable transformation" that exposes the scaling
     exponent alpha governing the unseen extreme regime.
  3. Score the candidate by the LEFT quadratic's fit quality plus the RIGHT
     log-log fit's residual; keep the best-scoring candidate.

The winning (xc_hat, y0_hat, alpha_hat, B_hat) is then reported directly as
"y0 + B*powv(x - xc, alpha)". Because this is the TRUE functional form (not a
local polynomial patch through the origin), the same closed form stays
correct arbitrarily far into the held-out congestion-cascade regime, unlike a
naive global power-law-through-the-origin fit that conflates both regimes
into one wrong exponent.
"""
import sys, math


def ols(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(v * v for v in xs)
    sxy = sum(a * b for a, b in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        slope = 0.0
    else:
        slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def solve3(A, b):
    D = det3(A)
    if abs(D) < 1e-9:
        return None
    cols = [[row[c] for row in A] for c in range(3)]
    out = []
    for c in range(3):
        M = [row[:] for row in A]
        for r in range(3):
            M[r][c] = b[r]
        out.append(det3(M) / D)
    return tuple(out)


def fit_quadratic(xs, ys):
    n = len(xs)
    S0 = float(n)
    S1 = sum(xs); S2 = sum(v * v for v in xs)
    S3 = sum(v ** 3 for v in xs); S4 = sum(v ** 4 for v in xs)
    T0 = sum(ys); T1 = sum(a * b for a, b in zip(xs, ys))
    T2 = sum(a * a * b for a, b in zip(xs, ys))
    A = [[S0, S1, S2], [S1, S2, S3], [S2, S3, S4]]
    b = [T0, T1, T2]
    return solve3(A, b)


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    xs_all, ys_all = [], []
    for _ in range(n):
        x = float(data[idx]); idx += 1
        y = float(data[idx]); idx += 1
        xs_all.append(x); ys_all.append(y)

    order = sorted(range(n), key=lambda i: xs_all[i])
    xs_all = [xs_all[i] for i in order]
    ys_all = [ys_all[i] for i in order]

    # candidate boundary points: interior deciles of the training range,
    # stepping through the sorted x values so every candidate is actually
    # attained by data on both sides.
    lo_i, hi_i = int(0.15 * n), int(0.85 * n)
    step = max(1, (hi_i - lo_i) // 40)
    candidates = list(range(lo_i, hi_i, step))

    best = None  # (score, xc, y0, alpha, B)
    for ci in candidates:
        xc_try = xs_all[ci]
        left_x = [x for x in xs_all if x < xc_try]
        left_y = [y for x, y in zip(xs_all, ys_all) if x < xc_try]
        right = [(x, y) for x, y in zip(xs_all, ys_all) if x >= xc_try]
        if len(left_x) < 10 or len(right) < 10:
            continue
        quad = fit_quadratic(left_x, left_y)
        if quad is None:
            continue
        c0, c1, c2 = quad
        y0 = c0 + c1 * xc_try + c2 * xc_try * xc_try
        if y0 <= 0.0:
            continue
        pos = [(x, y - y0) for x, y in right if x > xc_try and (y - y0) > 1e-9]
        if len(pos) < 8:
            continue
        xs_log = [math.log(x - xc_try) for x, _ in pos]
        ys_log = [math.log(r) for _, r in pos]
        alpha_try, log_B_try = ols(xs_log, ys_log)
        pred_log = [log_B_try + alpha_try * xl for xl in xs_log]
        sse_right = sum((p - a) ** 2 for p, a in zip(pred_log, ys_log)) / len(pos)
        sse_left = sum((y - (c0 + c1 * x + c2 * x * x)) ** 2
                        for x, y in zip(left_x, left_y)) / len(left_x)
        score = sse_right + sse_left / (y0 * y0 + 1e-9)
        if best is None or score < best[0]:
            best = (score, xc_try, y0, alpha_try, math.exp(log_B_try))

    if best is None:
        # fallback: behave like the global power-law recipe if the search
        # could not find any admissible split (should not happen in practice)
        xs = [math.log(v) for v in xs_all]
        ys = [math.log(v) for v in ys_all]
        p, log_C = ols(xs, ys)
        print("%.10g * powv(x, %.10g)" % (math.exp(log_C), p))
        return

    _, xc_hat, y0_hat, alpha_hat, B_hat = best
    alpha_hat = max(0.5, min(6.0, alpha_hat))
    B_hat = max(1e-6, min(1e6, B_hat))

    print("%.10g + %.10g*powv(x - (%.10g), %.10g)" % (y0_hat, B_hat, xc_hat, alpha_hat))


if __name__ == "__main__":
    main()
