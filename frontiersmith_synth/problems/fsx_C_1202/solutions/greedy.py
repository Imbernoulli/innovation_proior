# TIER: greedy
# The obvious recipe: this LOOKS like a straight line on log-log paper, so
# fit a plain power law y = A * n^(-alpha) by ordinary least squares on
# (log n, log y), pooling every visible point together. No floor term at
# all -- the textbook first move for "error vs. training size" data.
#
# This is a double trap: (1) it structurally forces the prediction to 0 as
# n grows, but the true error never drops below the irreducible floor, and
# (2) OLS over ALL points blends the steep early-burst exponent together
# with the true shallow asymptotic exponent, biasing alpha itself. Both
# errors compound catastrophically at the held-out scales (3x-2000x past
# training).
import sys, math


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.5")
        return
    m = int(data[0])
    vals = data[2:]
    ns = [float(vals[2 * i]) for i in range(m)]
    ys = [float(vals[2 * i + 1]) for i in range(m)]

    xs = [math.log(n) for n in ns]
    lys = [math.log(max(1e-9, y)) for y in ys]
    nA = len(xs)
    mx = sum(xs) / nA
    my = sum(lys) / nA
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, lys))
    slope = sxy / sxx if sxx > 1e-12 else -0.3
    intercept = my - slope * mx
    A = math.exp(intercept)
    alpha = -slope

    print("%.8f * n ** (%.8f)" % (A, -alpha))


if __name__ == "__main__":
    main()
