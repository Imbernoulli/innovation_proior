# TIER: strong
# The insight: the signal is in the SECOND moment, and the mechanism is not
# one smooth curve -- it is two physically different regimes (an elastic
# multiplicative power law, then an additive wear-rate law) that are pinned
# together at their INTERSECTION, the knee load x0. Recover x0 by grid search:
# for each candidate split, fit a power law A*x0^p to the light-load side and
# a line b + C*(x-x0) to the heavy-load side, but CONSTRAIN the line's
# intercept b to equal the power law's value AT x0 (continuity -- the two
# regimes' formulas must intersect there, not float independently). Pick the
# split that minimises total training residual, then emit the exact piecewise
# expression via the DSL's `step` (Heaviside) switch so it generalises past
# the training window instead of blending the two regimes into one guess.
import sys, math


def fit_power(xs, vs):
    lx = [math.log(max(1e-6, x)) for x in xs]
    lv = [math.log(max(1e-9, v)) for v in vs]
    n = len(lx)
    mx = sum(lx) / n
    mv = sum(lv) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - mv) for a, b in zip(lx, lv))
    p = sxy / sxx if sxx > 1e-9 else 1.0
    A = math.exp(mv - p * mx)
    return A, p


def main():
    data = sys.stdin.read().split()
    if len(data) < 4:
        print("2.0")
        return
    n_x, R, mu = int(data[0]), int(data[1]), float(data[2])
    idx = 4
    xs, vs = [], []
    for _ in range(n_x):
        x = float(data[idx]); idx += 1
        ss = 0.0
        for _ in range(R):
            y = float(data[idx]); idx += 1
            ss += (y - mu) ** 2
        v = max(1e-6, ss / R)
        xs.append(x)
        vs.append(v)

    order = sorted(range(n_x), key=lambda i: xs[i])
    sx = [xs[i] for i in order]
    sv = [vs[i] for i in order]

    best = None
    best_params = None
    # candidate knees: grid over the interior of the observed range
    lo_x, hi_x = sx[0], sx[-1]
    n_cand = 60
    # keep the knee away from the extreme ends of the observed range so both
    # sub-fits always have enough spread to be stable (avoids an unstable,
    # noise-driven slope on whichever side gets starved of points)
    for c in range(int(0.15 * n_cand), int(0.85 * n_cand) + 1):
        x0c = lo_x + (hi_x - lo_x) * c / n_cand
        left = [(x, v) for x, v in zip(sx, sv) if x < x0c]
        right = [(x, v) for x, v in zip(sx, sv) if x >= x0c]
        if len(left) < 6 or len(right) < 6:
            continue
        A, p = fit_power([x for x, _ in left], [v for _, v in left])
        b = A * (x0c ** p)
        # A per-load empirical variance estimate from only R repeats is
        # itself noisy, and that noise SCALES WITH the true variance being
        # estimated (sample-variance sampling error ~ v_true^2). Plain OLS
        # on the raw-unit right-side data lets a single unlucky big-v point
        # hijack the slope. Weight each point by 1/v_i^2 (inverse variance
        # of the noise on the variance estimate itself) -- a weighted-
        # least-squares correction for the SAME second-moment mechanism the
        # whole problem is about.
        wsum_xy = 0.0
        wsum_xx = 0.0
        for x, v in right:
            w = 1.0 / max(v, 1e-6) ** 2
            wsum_xy += w * (x - x0c) * (v - b)
            wsum_xx += w * (x - x0c) ** 2
        C = wsum_xy / wsum_xx if wsum_xx > 1e-12 else 0.0
        # the wear regime can only ADD wobble as load increases beyond the
        # knee; physically C >= 0. Clamping also guarantees the emitted
        # expression stays strictly positive at any x >= x0 (b > 0, C >= 0).
        C = max(C, 0.0)
        # model-selection criterion: WEIGHTED SSE (same 1/v^2 weighting) in
        # original variance units, evaluated on ALL training points.
        sse = 0.0
        for x, v in zip(sx, sv):
            if x < x0c:
                pred = A * (x ** p)
            else:
                pred = b + C * (x - x0c)
            w = 1.0 / max(v, 1e-6) ** 2
            sse += w * (pred - v) ** 2
        if best is None or sse < best:
            best = sse
            best_params = (x0c, A, p, C)

    if best_params is None:
        A, p = fit_power(sx, sv)
        print("%.9f * x ** %.9f" % (A, p))
        return

    x0, A, p, C = best_params
    b = A * (x0 ** p)
    # g(x) = (A*x^p)*(1-step(x-x0)) + (b + C*(x-x0))*step(x-x0)
    expr = ("(%.9f * x ** %.9f) * (1 - step(x - %.9f)) + "
            "(%.9f + %.9f * (x - %.9f)) * step(x - %.9f)") % (A, p, x0, b, C, x0, x0)
    print(expr)


if __name__ == "__main__":
    main()
