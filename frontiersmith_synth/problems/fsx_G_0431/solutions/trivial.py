# TIER: trivial
"""Emit the constant train-mean of y -- reproduces the grader's constant-mean
baseline, so it scores ~0.1."""
import sys


def main():
    toks = sys.stdin.read().split()
    vals = [float(t) for t in toks]
    ys = [vals[i + 4] for i in range(0, len(vals) - 4, 5)]
    m = sum(ys) / len(ys)
    sys.stdout.write("%.8f\n" % m)


if __name__ == "__main__":
    main()
