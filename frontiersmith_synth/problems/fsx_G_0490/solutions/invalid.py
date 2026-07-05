# TIER: invalid
"""Emit garbage (all zeros): rows/columns are not permutations -> not a Latin square -> Ratio 0."""
import sys


def main():
    data = sys.stdin.read().split()
    n, k = int(data[0]), int(data[1])
    out = []
    for _ in range(k):
        for _ in range(n):
            out.append(" ".join(["0"] * n))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
