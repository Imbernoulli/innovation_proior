# TIER: trivial
# Constant model: predict the mean training output, ignoring K and L entirely.
# Reproduces the grader's internal baseline -> Ratio ~ 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # header: n_train test_id ; then rows of K L y
    vals = data[2:]
    ys = [float(vals[3 * i + 2]) for i in range(n)]
    mean = sum(ys) / len(ys)
    sys.stdout.write("%r\n" % mean)


if __name__ == "__main__":
    main()
