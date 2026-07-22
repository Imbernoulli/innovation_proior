# TIER: invalid
"""Emits an infeasible artifact: concentrations that are negative and do not
sum to 1 -- must be rejected by the checker with Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    M = int(data[1])
    vals = [-0.3] * M
    vals[0] = 5.0
    print(" ".join(f"{v:.4f}" for v in vals))


if __name__ == "__main__":
    main()
