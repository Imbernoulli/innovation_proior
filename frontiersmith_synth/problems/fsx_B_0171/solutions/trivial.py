# TIER: trivial
# Baseline: predict a single constant = mean of the training targets.
# Matches the checker's internal constant-predictor baseline -> Ratio ~ 0.1.
import sys
import numpy as np


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    ys = []
    for _ in range(n):
        next(it); next(it); next(it); next(it)
        ys.append(float(next(it)))
    m = float(np.mean(ys))
    sys.stdout.write("%.8f\n" % m)


if __name__ == "__main__":
    main()
