# TIER: invalid
#!/usr/bin/env python3
"""Emits an out-of-range participation share (-1 for every layer) --
infeasible, must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    M = int(data[0])
    print(" ".join(["-1"] * M))


if __name__ == "__main__":
    main()
