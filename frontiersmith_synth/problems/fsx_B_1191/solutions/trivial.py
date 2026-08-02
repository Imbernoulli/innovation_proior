# TIER: trivial
"""Reproduces the checker's own internal baseline: predict a flat 40% of
nameplate capacity for every row, ignoring irradiance and temperature
entirely. Scores ~0.1 by construction."""
import sys


def main():
    data = sys.stdin.read().split()
    # header: n t N
    N = float(data[2])
    print("0.4 * N")


if __name__ == "__main__":
    main()
