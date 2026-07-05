# TIER: trivial
# Flat-persistence baseline: predict a constant equal to the mean of the last 3
# training y-values. This reproduces the checker's internal baseline B, so it
# scores ~0.1 by construction.
import sys


def main():
    toks = sys.stdin.read().split()
    m = int(toks[1])
    ys = []
    idx = 2
    for _ in range(m):
        ys.append(float(toks[idx + 1]))
        idx += 2
    const = sum(ys[-3:]) / 3.0
    sys.stdout.write("%.10g\n" % const)


if __name__ == "__main__":
    main()
