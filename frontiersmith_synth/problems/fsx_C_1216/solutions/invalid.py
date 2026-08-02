# TIER: invalid
"""Deliberately infeasible: a single accepting state that self-loops on every symbol.
Accepts every trace, including every forbidden one -- must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    m = int(data[0])
    print(1)
    print(1)
    print(" ".join("0" for _ in range(m)))


if __name__ == "__main__":
    main()
