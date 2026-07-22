# TIER: strong
# The insight: fit the obvious single-mechanism affine model first (like
# greedy) and note that its training fit already looks excellent -- the
# cubic term's contribution is smaller than the thermometer's own reading
# noise on every single logged row, so no ONE row's residual ever betrays
# it, and the training-error improvement from adding a cubic feature is too
# small to look worth the complexity by itself. Trusting that in-sample
# "goodness of fit" would stop right here (greedy does).
#
# Instead, commit to the physically-motivated ADDITIVE DECOMPOSITION stated
# in the problem (conduction + radiation) and jointly refit BOTH
# coefficients dT = a*(T-ambient) + b*(T-ambient)**3 by ordinary least
# squares over the ENTIRE log at once. By the Frisch-Waugh-Lovell identity
# this joint fit is exactly equivalent to first regressing out the dominant
# linear trend and reading the correctly-partialled residual signal against
# the cubic feature -- i.e. it IS "reading the systematic residual
# curvature", done the numerically rigorous way (aggregating every row,
# rather than eyeballing any one point's residual). Averaging over the
# whole log is what makes the small, per-row-invisible coefficient
# recoverable, and the recovered law now extrapolates correctly into the
# far-hotter held-out regime that the affine-only model cannot represent.
import sys


def ols1(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    p = sxy / sxx if sxx > 0 else 0.0
    q = my - p * mx
    return p, q


def ols2_noint(x1s, x2s, ys):
    """Joint least squares fit y = a*x1 + b*x2 (no intercept)."""
    s11 = sum(x * x for x in x1s)
    s22 = sum(x * x for x in x2s)
    s12 = sum(x1 * x2 for x1, x2 in zip(x1s, x2s))
    s1y = sum(x1 * y for x1, y in zip(x1s, ys))
    s2y = sum(x2 * y for x2, y in zip(x2s, ys))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-18:
        return 0.0, 0.0
    a = (s1y * s22 - s2y * s12) / det
    b = (s2y * s11 - s1y * s12) / det
    return a, b


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    ambient = float(data[2])
    Ts = [float(data[3 + 2 * i]) for i in range(n)]
    ds = [float(data[3 + 2 * i + 1]) for i in range(n)]
    xs = [T - ambient for T in Ts]

    # (Reference only, to show the trap: the affine-only fit greedy would stop at.)
    p1, q1 = ols1(xs, ds)

    x3 = [x ** 3 for x in xs]
    a_hat, b_hat = ols2_noint(xs, x3, ds)

    if a_hat == 0.0 and b_hat == 0.0:
        # Degenerate fit (should not happen on real data) -- fall back.
        print("( %.15g ) * ( T - ( %.10g ) ) + ( %.15g )" % (p1, ambient, q1))
    else:
        print("( %.15g ) * ( T - ( %.10g ) ) + ( %.15g ) * ( T - ( %.10g ) ) ** 3"
              % (a_hat, ambient, b_hat, ambient))


if __name__ == "__main__":
    main()
