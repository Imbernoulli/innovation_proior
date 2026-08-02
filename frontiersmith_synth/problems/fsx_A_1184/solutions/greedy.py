# TIER: greedy
import sys, math


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    testId = int(next(it)); T = int(next(it)); M = int(next(it))
    sigma = float(next(it)); thr = float(next(it)); lam = float(next(it))
    r = [int(next(it)) for _ in range(M)]
    trace = [float(next(it)) for _ in range(T)]

    # Symmetric independent matched filter: the obvious first attempt.
    # Assume every peak is a plain (tau=0) symmetric Gaussian and estimate each
    # slot's area purely from a local window around its own nominal retention
    # time -- no cross-slot deconvolution, no shared tailing parameter.
    w = 1.5 * sigma
    x = []
    for ri in r:
        lo = max(0, int(math.floor(ri - w)))
        hi = min(T, int(math.ceil(ri + w)) + 1)
        x.append(sum(trace[lo:hi]))

    print(" ".join("%.6f" % v for v in x))


main()
