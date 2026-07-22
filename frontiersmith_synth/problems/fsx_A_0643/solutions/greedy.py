# TIER: greedy
"""
Greedy / textbook approach: a single power-law fit.

Cd = A' * Re^p'  is linear in log-log space: log(Cd) = log(A') + p'*log(Re).
Ordinary least squares over ALL training rows gives an excellent in-sample
fit (the dominant mechanism explains almost everything inside the flume's
reachable window) -- but a single power law has only one exponent, so it
cannot represent the crossover to the shallower subdominant mechanism that
takes over far outside the window. Extrapolated onto the held-out grid it
decays at the wrong (too steep) rate.
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
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    Res, Cds = [], []
    for _ in range(n):
        Re = float(data[idx]); idx += 1
        Cd = float(data[idx]); idx += 1
        Res.append(Re); Cds.append(Cd)

    xs = [math.log(r) for r in Res]
    ys = [math.log(c) for c in Cds]
    p, log_A = ols(xs, ys)
    A = math.exp(log_A)

    print("%.10g * powv(Re, %.10g)" % (A, p))


if __name__ == "__main__":
    main()
