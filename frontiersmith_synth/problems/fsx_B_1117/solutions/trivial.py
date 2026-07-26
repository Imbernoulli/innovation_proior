# TIER: trivial
# Do-nothing baseline: predict the mean of the training rates as a CONSTANT
# function of (S, C). This is exactly the checker's own internal baseline
# construction -> reproduces Ratio ~= 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("1.0")
        return
    n_regimes = int(data[1])
    n_pts = int(data[2])
    vals = data[3:]
    n = n_regimes * n_pts
    total = 0.0
    for i in range(n):
        total += float(vals[3 * i + 2])
    mean_rate = total / n if n else 1.0
    print("%.6f" % mean_rate)


if __name__ == "__main__":
    main()
