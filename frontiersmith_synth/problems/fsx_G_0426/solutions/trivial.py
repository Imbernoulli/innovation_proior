# TIER: trivial
# Baseline: predict a single constant = the mean of the training absolute
# magnitudes.  Reproduces the grader's internal constant baseline -> ~0.1.
import sys


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    ys = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 4:
            ys.append(float(p[3]))
    m = sum(ys) / len(ys) if ys else 0.0
    sys.stdout.write(repr(m) + "\n")


if __name__ == "__main__":
    main()
