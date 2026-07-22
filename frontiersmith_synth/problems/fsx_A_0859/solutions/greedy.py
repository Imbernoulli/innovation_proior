# TIER: greedy
# The obvious recipe: notice the wobble clearly grows with load, so fit ONE
# smooth power law sigma^2(x) = A * x^p to the (load, empirical-variance)
# pairs by ordinary least squares in log-log space, across ALL the training
# data at once. This tracks the training window reasonably (it is, after all,
# a monotone-growing curve through mixed-regime data) but a single power-law
# shape cannot represent the mechanism's regime SWITCH -- the additive wear
# term beyond the knee -- so it extrapolates to heavier loads on a smooth
# continuation of the wrong curvature.
import sys, math


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

    # OLS fit of ln(v) = ln(A) + p*ln(x)
    lx = [math.log(max(1e-6, x)) for x in xs]
    lv = [math.log(v) for v in vs]
    n = len(lx)
    mx = sum(lx) / n
    mv = sum(lv) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - mv) for a, b in zip(lx, lv))
    p = sxy / sxx if sxx > 1e-12 else 1.0
    lnA = mv - p * mx
    A = math.exp(lnA)

    print("%.9f * x ** %.9f" % (A, p))


if __name__ == "__main__":
    main()
