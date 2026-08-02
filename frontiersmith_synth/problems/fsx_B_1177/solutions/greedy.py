# TIER: greedy
# The obvious recipe: assume the whole visible curve is ONE smooth continuous
# mode and fit a generic power law v = c * f^p by ordinary log-log linear
# regression over ALL training points (ignores the CA/CB/CD/CE branch
# constants given in the input even though they are right there -- a curve
# fit doesn't need them, it just needs the (f, v) pairs). This reproduces the
# visible branch beautifully (training is always a SINGLE branch) but keeps
# extrapolating the SAME shape forever, so it silently rides straight through
# any mode crossing instead of switching to the other branch.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.0"); return
    n = int(data[0])
    rows = data[6:]
    freqs = [float(rows[2 * i]) for i in range(n)]
    vals = [float(rows[2 * i + 1]) for i in range(n)]

    xs = [math.log(f) for f in freqs]
    ys = [math.log(max(v, 1e-6)) for v in vals]
    m = len(xs)
    mx = sum(xs) / m
    my = sum(ys) / m
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    p = sxy / sxx if sxx > 1e-12 else 0.5
    c = math.exp(my - p * mx)

    print("%.6f * f ** %.6f" % (c, p))


if __name__ == "__main__":
    main()
