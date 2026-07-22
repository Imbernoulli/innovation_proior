# TIER: strong
# Structural recovery via a REFORMULATION, not "more grid points": the
# statement tells us the response is a SEPARABLE product of one univariate
# SATURATING curve in amplitude and one univariate RESONANT (peaked) curve
# in frequency:
#     R(f,a) = G * (As*tanh(a/As)) / (1 + ((f-f0)/w)**2)
# That means for ANY candidate shape parameters (f0, w, As), the transformed
# feature z(f,a) = tanh(a/As) / (1+((f-f0)/w)**2) turns the WHOLE nonlinear
# surface into a SINGLE zero-intercept linear regressor y ~ p*z (the linear
# gain drops out via ordinary 1-D least squares) -- so only the three SHAPE
# parameters need a search, not four independent unknowns.
#
# The trap a naive version of this search falls into: a too-NARROW candidate
# w turns z into a near-spike that can chase small-signal noise with a huge
# compensating gain p (classic overfit -- low in-sample error, catastrophic
# extrapolation). The insight that separates this from "grid search harder":
# among all candidates whose in-sample fit is statistically indistinguishable
# from the best one found, prefer the LARGEST w (an Occam's-razor
# regularizer -- the simplest, least-spiky resonance consistent with the
# data) rather than the raw minimizer. This is what a quadratic-surface
# recipe (greedy) has no notion of at all: it isn't searching a shape
# family, so there's nothing to regularize toward.
import sys


def _tanh(x):
    if x > 40.0:
        return 1.0
    if x < -40.0:
        return -1.0
    e2x = pow(2.718281828459045, 2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


def fit_gain(rows, f0, w, As):
    """Zero-intercept 1-D OLS of y on z(f,a) = tanh(a/As)/(1+((f-f0)/w)**2)."""
    szz = 0.0
    szy = 0.0
    zs = []
    for f, a, y in rows:
        z = _tanh(a / As) / (1.0 + ((f - f0) / w) ** 2)
        zs.append(z)
        szz += z * z
        szy += z * y
    if szz < 1e-9:
        return None
    p = szy / szz
    sse = 0.0
    for (f, a, y), z in zip(rows, zs):
        d = p * z - y
        sse += d * d
    return sse, p


def grid_search(rows, f0_vals, w_vals, As_vals):
    out = []
    for f0 in f0_vals:
        for w in w_vals:
            for As in As_vals:
                res = fit_gain(rows, f0, w, As)
                if res is None:
                    continue
                sse, p = res
                out.append((sse, p, f0, w, As))
    return out


def lin(lo, hi, nsteps):
    if nsteps <= 1:
        return [0.5 * (lo + hi)]
    return [lo + (hi - lo) * i / (nsteps - 1) for i in range(nsteps)]


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    rows = []
    idx = 1
    for _ in range(n):
        f = float(toks[idx]); a = float(toks[idx + 1]); y = float(toks[idx + 2])
        idx += 3
        rows.append((f, a, y))

    # coarse pass over the shape-parameter volume (generic wide priors, not
    # peeking at the generator's exact draw ranges)
    f0_vals = lin(4.0, 26.0, 17)
    w_vals = lin(0.4, 5.0, 11)
    As_vals = lin(1.0, 10.0, 11)
    cands = grid_search(rows, f0_vals, w_vals, As_vals)
    if not cands:
        print("3.0*tanh(a/3.0)/(1+((f-12.0)/2.0)**2)")
        return
    cands.sort(key=lambda c: c[0])
    _, _, f0, w, As = cands[0]

    # local refinement pass around the coarse optimum
    df0 = f0_vals[1] - f0_vals[0]
    dw = w_vals[1] - w_vals[0]
    dAs = As_vals[1] - As_vals[0]
    f0_ref = lin(f0 - df0, f0 + df0, 7)
    w_ref = lin(max(0.2, w - dw), w + dw, 7)
    As_ref = lin(max(0.3, As - dAs), As + dAs, 7)
    cands += grid_search(rows, f0_ref, w_ref, As_ref)
    cands.sort(key=lambda c: c[0])

    # Occam's-razor regularization: among candidates within 15% of the best
    # in-sample SSE, take the one with the LARGEST resonance width w --
    # this is what rules out a spiky narrow-w overfit that happens to shave
    # a sliver off the training error at the cost of catastrophic
    # extrapolation.
    best_sse = cands[0][0]
    tol = best_sse * 1.15
    plausible = [c for c in cands if c[0] <= tol]
    plausible.sort(key=lambda c: -c[3])
    _, p, f0, w, As = plausible[0]

    print("%.10g*tanh(a/%.10g)/(1+((f-%.10g)/%.10g)**2)" % (p, As, f0, w))


if __name__ == "__main__":
    main()
