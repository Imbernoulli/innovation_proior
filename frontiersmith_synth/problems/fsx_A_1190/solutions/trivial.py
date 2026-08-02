# TIER: trivial
"""Degenerate baseline: declare all K endmembers equal to the mean pixel spectrum, and
give every pixel the uniform abundance vector (1/K,...,1/K). Feasible (nonneg, sums to
1, reconstructs the mean pixel for everyone) but throws away all structure."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    _t = int(next(it))
    R = int(next(it)); K = int(next(it)); N = int(next(it))
    Y = [[float(next(it)) for _ in range(R)] for _ in range(N)]

    mean_y = [sum(Y[j][r] for j in range(N)) / N for r in range(R)]

    out = []
    for _k in range(K):
        out.append(" ".join("%.6f" % v for v in mean_y))
    for _j in range(N):
        out.append(" ".join("%.6f" % (1.0 / K) for _ in range(K)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
