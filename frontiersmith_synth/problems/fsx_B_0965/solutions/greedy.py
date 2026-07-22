# TIER: greedy
# The obvious first move: the data ARRIVES as a (tag number n, stamped value) time series, so
# fit a straight line in n by ordinary least squares on the training rows and extrapolate:
# f(n) ~= c*n + d.  This captures the AVERAGE multiplicative discount across the training
# population (a single global slope) but cannot see that the discount actually depends on
# WHICH primes divide n (a residue-class rule) or on how many times a small prime like 2
# divides n -- it treats every tag alike regardless of its factorization, so it is far off for
# any withheld tag whose factor structure departs from the population average (primes vs.
# highly-composite numbers land on opposite sides of the fitted line).
import sys


def main():
    toks = sys.stdin.read().split()
    ntr = int(toks[0])
    idx = 1
    pts = []
    for _ in range(ntr):
        n = int(toks[idx]); obs = int(toks[idx + 1])
        idx += 2
        pts.append((n, obs))

    N = len(pts)
    sx = sum(n for n, _ in pts)
    sy = sum(o for _, o in pts)
    sxx = sum(n * n for n, _ in pts)
    sxy = sum(n * o for n, o in pts)
    denom = N * sxx - sx * sx
    if denom == 0:
        c, d = 1.0, 0.0
    else:
        c = (N * sxy - sx * sy) / denom
        d = (sy - c * sx) / N

    print("MODE N")
    print("%.10g * n + %.10g" % (c, d))


if __name__ == "__main__":
    main()
