# TIER: invalid
"""Deliberately infeasible: races through every segment in a fixed tiny
duration, which for any real instance implies a cooling rate far above
rate_max (and a huge rate jump from rest) -- must be rejected by the
checker."""
import sys


def main():
    data = sys.stdin.read().split()
    M = int(data[0])
    t = [0.000001] * M
    print(" ".join(f"{x:.9f}" for x in t))


if __name__ == "__main__":
    main()
