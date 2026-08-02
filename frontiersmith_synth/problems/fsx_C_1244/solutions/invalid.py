# TIER: invalid
"""Emits an obviously non-permutation output (all cells crammed onto slot 0):
duplicate slots everywhere, must score 0 under strict feasibility checking."""
import sys


def main():
    data = sys.stdin.read().split()
    n_cells = int(data[0])
    print(" ".join("0" for _ in range(n_cells)))


if __name__ == "__main__":
    main()
