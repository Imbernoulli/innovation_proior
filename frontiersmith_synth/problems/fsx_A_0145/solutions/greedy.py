# TIER: greedy
"""Flat / constant emission profile -> c1 == 2 exactly, beating the triangle
baseline. A different, n-independent objective value from the other tiers."""
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    f = [1] * n
    sys.stdout.write(" ".join(map(str, f)) + "\n")


if __name__ == "__main__":
    main()
