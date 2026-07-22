# TIER: trivial
# Do-nothing baseline: predict a flat constant delta, the mean of the logged
# dT column.  This is EXACTLY the checker's own internal baseline construction
# -> reproduces Ratio ~= 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    deltas = [float(data[3 + 2 * i + 1]) for i in range(n)]
    mean_delta = sum(deltas) / len(deltas)
    print("( %.15g )" % mean_delta)


if __name__ == "__main__":
    main()
