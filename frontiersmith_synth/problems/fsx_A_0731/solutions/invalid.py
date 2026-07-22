# TIER: invalid
#!/usr/bin/env python3
"""Emits negative levee heights -- infeasible, must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    print(" ".join(["-1"] * N))


if __name__ == "__main__":
    main()
