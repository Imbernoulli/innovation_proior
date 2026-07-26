# TIER: strong
# Insight: recover the MECHANISM, not the surface -- and fit it JOINTLY across
# every logged nutrient level at once, not one p-group at a time.  A per-p
# decomposition into (K_p, r_p) is ill-posed here: with only 10 time points
# per p and a shallow early-saturation regime, K and r trade off along a long
# near-flat ridge (K*r*t is the identifiable quantity, not K and r alone), so
# an isolated per-p fit can drift to a degenerate corner.  Pooling all 5
# nutrient levels' rows and fitting the SHARED four constants (K0, alpha, r0,
# beta) of the assumed allometric saturation law S=K0*p**alpha*(1-exp(-r0*
# (1-p)**beta*t)) breaks that ridge: different p rows constrain the SAME
# alpha, beta pair, so the cross-p variation itself pins the exponents.  For
# fixed (alpha, beta, r0) the amplitude K0 is the closed-form least-squares
# slope of S against x=p**alpha*(1-exp(-r0*(1-p)**beta*t)); r0 is found by a
# bounded 1-D golden-section search (K0 solved in closed form at each trial);
# alpha and beta are found by an outer coordinate-descent of bounded golden
# searches.  Because this is the TRUE functional family (not a local
# polynomial patch), the emitted closed form extrapolates correctly past the
# training band.  The coarse recovery + irreducible held-out noise keep the
# score below 1.0.
import sys
import math


def golden_min(f, lo, hi, iters=40):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - gr * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (b - a)
            f2 = f(x2)
    return (a + b) / 2.0


def sse_given_r0(alpha, beta, r0, rows):
    xs = []
    ss = []
    for t, p, s in rows:
        u = p ** alpha
        v = (1.0 - p) ** beta
        x = u * (1.0 - math.exp(-r0 * v * t))
        xs.append(x)
        ss.append(s)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * s for x, s in zip(xs, ss))
    k0 = sxy / sxx if sxx > 1e-12 else 0.0
    sse = sum((s - k0 * x) ** 2 for s, x in zip(ss, xs))
    return sse, k0


def best_over_r0(alpha, beta, rows):
    r0 = golden_min(lambda r: sse_given_r0(alpha, beta, r, rows)[0], 0.02, 1.5)
    sse, k0 = sse_given_r0(alpha, beta, r0, rows)
    return sse, k0, r0


def fit_joint(rows):
    alpha, beta = 0.5, 0.5
    for _ in range(5):
        alpha = golden_min(lambda a: best_over_r0(a, beta, rows)[0], 0.05, 2.0)
        beta = golden_min(lambda b: best_over_r0(alpha, b, rows)[0], 0.05, 2.0)
    sse, k0, r0 = best_over_r0(alpha, beta, rows)
    return k0, alpha, r0, beta


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    rows = []
    for ln in data[1:1 + n]:
        parts = ln.split()
        if len(parts) >= 3:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))

    K0, alpha, r0, beta = fit_joint(rows)

    expr = "%r*(p**%r)*(1-exp(-%r*((1-p)**%r)*t))" % (K0, alpha, r0, beta)
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()
