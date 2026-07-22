# TIER: strong
"""
Residual curvature as a hypothesis test, then a reformulation that exploits
the harmonic (resistors-in-series) structure of the two channels.

Step 1 (hypothesis test): bin the training rows by T, fit BOTH a straight
line and a quadratic to log(r) vs 1/T on the binned points. In this family a
single Arrhenius channel is a straight line in these coordinates; if the
data really were one mechanism, adding a quadratic term should barely help.
A systematic, non-random curvature -- even though it is small enough that
the straight-line R^2 still looks excellent -- is evidence that a second,
competing channel is present. Only escalate to the two-channel model when
the quadratic term earns its keep (meaningfully reduces the fit error); this
guards against manufacturing structure out of pure noise on an easy/flat
instance.

Step 2 (reformulation): the two channels combine as a harmonic mean,
r = k1*k2/(k1+k2), i.e. 1/r = 1/k1 + 1/k2. That reciprocal is LINEAR in the
two coefficients c1 = 1/A1, c2 = 1/A2 for any FIXED pair of characteristic
temperatures (theta1, theta2):

    1/r  ==  c1 * exp(theta1/T)  +  c2 * exp(-theta2/T)

so for each candidate (theta1, theta2) the best (c1, c2) is a closed-form
2x2 least-squares solve. A grid search over (theta1, theta2), refined by
local coordinate descent, finds the pair that minimizes the residual sum of
squares on the (noise-reduced, binned) reciprocal-rate data. This carries
the correct high-T turnover and low-T collapse into the held-out regimes
that a single Arrhenius channel cannot represent.
"""
import sys, math


def ols(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, (sy / n if n else 0.0)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def quad_fit_sse(xs, ys):
    """Least-squares quadratic y = a + b*x + c*x^2; returns SSE."""
    n = len(xs)
    S0, S1, S2, S3, S4 = n, sum(xs), sum(x * x for x in xs), \
        sum(x ** 3 for x in xs), sum(x ** 4 for x in xs)
    T0 = sum(ys)
    T1 = sum(x * y for x, y in zip(xs, ys))
    T2 = sum((x * x) * y for x, y in zip(xs, ys))
    # solve 3x3 normal equations by Cramer's rule
    M = [[S0, S1, S2], [S1, S2, S3], [S2, S3, S4]]
    V = [T0, T1, T2]

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    D = det3(M)
    if abs(D) < 1e-30:
        return None
    coeffs = []
    for col in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][col] = V[r]
        coeffs.append(det3(Mi) / D)
    a, b, c = coeffs
    sse = sum((a + b * x + c * x * x - y) ** 2 for x, y in zip(xs, ys))
    return sse


def bin_average(Ts, Rs, n_bins):
    n = len(Ts)
    order = sorted(range(n), key=lambda i: Ts[i])
    Ts = [Ts[i] for i in order]; Rs = [Rs[i] for i in order]
    bins = [[] for _ in range(n_bins)]
    for i in range(n):
        b = min(n_bins - 1, i * n_bins // n)
        bins[b].append((Ts[i], Rs[i]))
    outT, outR = [], []
    for b in bins:
        if not b:
            continue
        Tm = sum(x for x, _ in b) / len(b)
        rm = math.exp(sum(math.log(r) for _, r in b) / len(b))
        outT.append(Tm); outR.append(rm)
    return outT, outR


def fit_c1c2(Ts, ys, theta1, theta2):
    """y = c1*exp(theta1/T) + c2*exp(-theta2/T); closed-form 2x2 solve."""
    s11 = s12 = s22 = b1 = b2 = 0.0
    for T, y in zip(Ts, ys):
        e1 = math.exp(theta1 / T); e2 = math.exp(-theta2 / T)
        s11 += e1 * e1; s12 += e1 * e2; s22 += e2 * e2
        b1 += e1 * y; b2 += e2 * y
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-30:
        return None, None, float("inf")
    c1 = (b1 * s22 - b2 * s12) / det
    c2 = (s11 * b2 - s12 * b1) / det
    sse = sum((c1 * math.exp(theta1 / T) + c2 * math.exp(-theta2 / T) - y) ** 2
               for T, y in zip(Ts, ys))
    return c1, c2, sse


def two_channel_search(binT, binY):
    best = (None, None, None, None, float("inf"))
    theta1_grid = [1200.0 + 30.0 * i for i in range(101)]   # 1200..4200
    theta2_grid = [300.0 + 30.0 * i for i in range(68)]     # 300..2310
    for th1 in theta1_grid:
        for th2 in theta2_grid:
            c1, c2, sse = fit_c1c2(binT, binY, th1, th2)
            if c1 is None or c1 <= 0.0 or c2 <= 0.0:
                continue
            if sse < best[4]:
                best = (th1, th2, c1, c2, sse)
    th1, th2, c1, c2, sse = best
    if th1 is None:
        return None
    step1 = step2 = 30.0
    for _ in range(8):
        improved = False
        for d1 in range(-3, 4):
            for d2 in range(-3, 4):
                if d1 == 0 and d2 == 0:
                    continue
                t1 = th1 + d1 * step1
                t2 = th2 + d2 * step2
                if t1 <= 0.0 or t2 <= 0.0:
                    continue
                c1n, c2n, ssen = fit_c1c2(binT, binY, t1, t2)
                if c1n is None or c1n <= 0.0 or c2n <= 0.0:
                    continue
                if ssen < sse:
                    th1, th2, c1, c2, sse = t1, t2, c1n, c2n, ssen
                    improved = True
        step1 *= 0.5; step2 *= 0.5
        if step1 < 0.2 and step2 < 0.2:
            break
    return th1, th2, c1, c2


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    Ts, Rs = [], []
    for _ in range(n):
        T = float(data[idx]); idx += 1
        r = float(data[idx]); idx += 1
        Ts.append(T); Rs.append(r)

    binT, binR = bin_average(Ts, Rs, 30)
    xs = [1.0 / T for T in binT]
    ys_lin = [math.log(r) for r in binR]

    slope, intercept = ols(xs, ys_lin)
    sse_lin = sum((slope * x + intercept - y) ** 2 for x, y in zip(xs, ys_lin))
    sse_quad = quad_fit_sse(xs, ys_lin)

    escalate = (sse_quad is not None and sse_lin > 0.0 and
                (sse_lin - sse_quad) / sse_lin > 0.005)

    if escalate:
        binY = [1.0 / r for r in binR]     # reciprocal-rate reformulation
        result = two_channel_search(binT, binY)
    else:
        result = None

    if result is None:
        theta = -slope
        A = math.exp(intercept)
        print("%.10g * expv(-%.10g / T)" % (A, theta))
        return

    th1, th2, c1, c2 = result
    A1 = 1.0 / c1
    A2 = 1.0 / c2
    # k1 = A1*expv(-th1/T); k2 = A2*expv(th2/T); r = k1*k2/(k1+k2)
    print("(%.10g * expv(-%.10g / T)) * (%.10g * expv(%.10g / T)) / "
          "((%.10g * expv(-%.10g / T)) + (%.10g * expv(%.10g / T)))"
          % (A1, th1, A2, th2, A1, th1, A2, th2))


if __name__ == "__main__":
    main()
