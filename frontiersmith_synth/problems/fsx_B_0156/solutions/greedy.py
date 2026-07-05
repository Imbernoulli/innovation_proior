# TIER: greedy
# Semilog model: fit eta = p + q*log(x) by least squares on the train points.
# Captures the increasing-with-scale trend better than a constant, but the
# wrong functional shape (unbounded log growth) mis-extrapolates at large x.
import sys
import math


def main():
    toks = sys.stdin.read().split()
    xs = [float(toks[i]) for i in range(0, len(toks), 2)]
    ys = [float(toks[i]) for i in range(1, len(toks), 2)]
    n = len(xs)
    fx = [math.log(x) for x in xs]
    sx = sum(fx)
    sy = sum(ys)
    sxx = sum(v * v for v in fx)
    sxy = sum(fx[i] * ys[i] for i in range(n))
    det = n * sxx - sx * sx
    q = (n * sxy - sx * sy) / det
    p = (sy - q * sx) / n
    print("%.10g + (%.10g)*log(x)" % (p, q))


if __name__ == "__main__":
    main()
