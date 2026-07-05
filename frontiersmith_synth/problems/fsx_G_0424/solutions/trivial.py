# TIER: trivial
# Baseline: predict a single constant = the mean of the training drag readings.
# Reproduces the grader's internal baseline -> ~0.1.
import sys


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    ys = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 3:
            ys.append(float(p[2]))
    m = sum(ys) / len(ys) if ys else 0.0
    sys.stdout.write(repr(m) + "\n")


if __name__ == "__main__":
    main()
