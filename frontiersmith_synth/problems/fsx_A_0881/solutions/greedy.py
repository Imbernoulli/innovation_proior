# TIER: greedy
# The obvious recipe: the log LOOKS like a straight line (the cubic term is a
# minor correction there), so fit a single-mechanism affine model
# dT = p*T + q by ordinary least squares directly on the logged rows and
# report it. This nails the logged range but has no way to represent a
# second, cubic regime, so it collapses on the much hotter held-out range
# where that regime dominates.
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


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    Ts = [float(data[3 + 2 * i]) for i in range(n)]
    ds = [float(data[3 + 2 * i + 1]) for i in range(n)]

    p, q = ols1(Ts, ds)
    print("( %.15g ) * T + ( %.15g )" % (p, q))


if __name__ == "__main__":
    main()
