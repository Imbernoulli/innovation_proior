# TIER: trivial
# Constant model: predict the mean of the training efficiencies everywhere.
# Reproduces the checker's internal baseline -> Ratio ~= 0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    ys = [float(toks[i]) for i in range(1, len(toks), 2)]
    m = sum(ys) / len(ys)
    print("%.10g" % m)


if __name__ == "__main__":
    main()
