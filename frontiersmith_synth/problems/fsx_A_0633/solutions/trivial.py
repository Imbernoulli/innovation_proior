# TIER: trivial
"""Reproduces the checker's own baseline-shaped idea: the smallest possible
valid touch, a single out-and-back swap.  No musical or palindrome awareness."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    # We don't even need to look at n/Kmax/musical rows -- swapping positions
    # 1,2 out and back is always valid for n>=2 and Kmax>=2.
    print(2)
    print(1)
    print(1)


if __name__ == "__main__":
    main()
