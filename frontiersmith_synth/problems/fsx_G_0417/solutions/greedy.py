# TIER: greedy
"""Free-exponent single power-law fit: F = A * r**(-p), with (ln A, p) by exact
least squares on (ln r, ln F).  This captures the effective inner-band slope, so it
beats the fixed inverse-square baseline -- but a single power law cannot represent
the leading term plus its short-range correction, so it extrapolates the wrong slope
into the outer band -> a middling score."""
import sys, math


def main():
    vals = [float(t) for t in sys.stdin.read().split()]
    rows = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    xs = []
    ys = []
    for (r, f) in rows:
        if f > 0.0 and r > 0.0:
            xs.append(math.log(r))
            ys.append(math.log(f))
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else -2.0
    intercept = my - slope * mx
    A = math.exp(intercept)
    p = -slope
    sys.stdout.write("%.10f * r**(%.10f)\n" % (A, -p))


if __name__ == "__main__":
    main()
