# TIER: invalid
"""Deliberately infeasible: releases nothing at all, every week, for the whole horizon.
Storage only ever grows, so it overflows capacity in the very first scenario within a
handful of weeks -- must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    print(" ".join(["0"] * T))


if __name__ == "__main__":
    main()
