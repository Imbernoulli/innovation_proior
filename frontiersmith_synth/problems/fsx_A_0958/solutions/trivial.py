# TIER: trivial
# Constant model: predict the mean of the training response everywhere.
# Reproduces the checker's internal baseline -> Ratio ~= 0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    ys = [float(toks[1 + 3 * i + 2]) for i in range(n)]
    m = sum(ys) / len(ys)
    print("%.10g" % m)


if __name__ == "__main__":
    main()
