# TIER: trivial
"""Fully-immunized baseline: take zero risk, every year, every bucket.
This is exactly the construction the checker uses as its own baseline B,
so this solution reproduces Ratio ~= 0.1 by definition."""
import sys


def main():
    toks = sys.stdin.read().split()
    T = int(toks[0])
    lines = ["0.0 0.0 0.0 0.0 0.0"] * T
    print("\n".join(lines))


if __name__ == "__main__":
    main()
