# TIER: trivial
"""Constant-mean predictor: emit the mean of the training y as a bare number.
This reproduces the grader's internal baseline -> Ratio ~ 0.1."""
import sys


def main():
    vals = [float(t) for t in sys.stdin.read().split()]
    ys = [vals[i + 4] for i in range(0, len(vals), 5)]
    mean_y = sum(ys) / len(ys)
    sys.stdout.write("%.10f\n" % mean_y)


if __name__ == "__main__":
    main()
