# TIER: trivial
# Do-nothing: predict a flat constant equal to the mean of the observed
# sub-critical census.  This is exactly the checker's internal baseline
# construction, so it reproduces Ratio ~ 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[1])
    vals = [float(data[3 + 2 * i + 1]) for i in range(n)]
    mean_s = sum(vals) / len(vals) if vals else 0.0
    print("%.8f" % mean_s)


if __name__ == "__main__":
    main()
