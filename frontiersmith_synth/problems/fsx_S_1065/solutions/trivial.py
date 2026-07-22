# TIER: trivial
"""Demolish piers in the order they are labeled in the input. This is exactly the
checker's own internal baseline construction -- it ignores the brace pattern
entirely, so it always reproduces the reference baseline score."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    print(" ".join(str(i) for i in range(1, n + 1)))


if __name__ == "__main__":
    main()
