# TIER: trivial
"""Predict the constant mean of the training displacements. Reproduces the
checker's own internal baseline exactly (a single constant is 1 AST node,
same as the baseline's assumed penalty) -> Ratio ~= 0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    n_train = int(data[0])
    # header is "n_train test_id", then n_train rows of "F r y"
    ys = []
    idx = 2
    for _ in range(n_train):
        F = float(data[idx]); r = float(data[idx + 1]); y = float(data[idx + 2])
        idx += 3
        ys.append(y)
    mean_y = sum(ys) / len(ys) if ys else 0.0
    print("%.6f" % mean_y)


if __name__ == "__main__":
    main()
