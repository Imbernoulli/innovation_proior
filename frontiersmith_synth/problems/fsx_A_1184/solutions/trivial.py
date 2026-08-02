# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    testId = int(next(it)); T = int(next(it)); M = int(next(it))
    sigma = float(next(it)); thr = float(next(it)); lam = float(next(it))
    r = [int(next(it)) for _ in range(M)]
    trace = [float(next(it)) for _ in range(T)]

    # blind uniform baseline: declare every reference slot present, split the
    # total observed mass evenly across all M slots (ignores WHERE the mass is).
    total = sum(trace)
    x = total / M
    print(" ".join("%.6f" % x for _ in range(M)))


main()
