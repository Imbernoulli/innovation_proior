# TIER: greedy
"""
Greedy / textbook approach: the data visibly curves upward with increasing
steepness, so fit a single global power law through the origin,

    y = C * x^p        <=>   log(y) = log(C) + p*log(x)

by ordinary least squares in log-log space over ALL training rows, ignoring
any possible regime change. This "looks like scaling analysis" and tracks
the training band reasonably (both regimes are, after all, monotonically
increasing), but it never tests for -- let alone locates -- a regime
boundary: it silently assumes the power law runs through x=0 rather than
through the TRUE congestion threshold xc. That misattributes curvature from
the calm regime into the fitted exponent p, which then does not match the
true cascade exponent alpha and extrapolates at the wrong rate far past the
training band.
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


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    xs_raw, ys_raw = [], []
    for _ in range(n):
        x = float(data[idx]); idx += 1
        y = float(data[idx]); idx += 1
        xs_raw.append(x); ys_raw.append(y)

    xs = [math.log(v) for v in xs_raw]
    ys = [math.log(v) for v in ys_raw]
    p, log_C = ols(xs, ys)
    C = math.exp(log_C)

    print("%.10g * powv(x, %.10g)" % (C, p))


if __name__ == "__main__":
    main()
