# TIER: greedy
"""
Greedy / textbook approach: a single Arrhenius fit.

r = A' * exp(-theta' / T) is linear in Arrhenius coordinates:
log(r) = log(A') - theta'/T. Ordinary least squares of log(r) against 1/T
over ALL training rows gives an excellent in-sample fit (R^2 well above
0.99 -- inside the proofing window a single exponential explains almost
everything) -- but a single Arrhenius channel has only one characteristic
temperature, so it cannot represent the crossover to the competing
stability channel that bottlenecks the rate far outside the window. Blindly
extrapolated, it keeps climbing monotonically into the oven-overshoot
regime (where the true rate has already turned over and stalled) and falls
off too fast into the fridge-retard regime.
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
    Ts, Rs = [], []
    for _ in range(n):
        T = float(data[idx]); idx += 1
        r = float(data[idx]); idx += 1
        Ts.append(T); Rs.append(r)

    xs = [1.0 / T for T in Ts]
    ys = [math.log(r) for r in Rs]
    slope, intercept = ols(xs, ys)   # slope = -theta'
    theta = -slope
    A = math.exp(intercept)

    print("%.10g * expv(-%.10g / T)" % (A, theta))


if __name__ == "__main__":
    main()
