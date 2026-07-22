# TIER: trivial
"""Constant predictor: emit the mean of the training years.  Reproduces the
checker's own internal baseline construction (Ratio ~= 0.1)."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = [float(v) for v in data[2:2 + n]]
    mean_v = sum(vals) / len(vals)
    print("OUT %.6f" % mean_v)


if __name__ == "__main__":
    main()
