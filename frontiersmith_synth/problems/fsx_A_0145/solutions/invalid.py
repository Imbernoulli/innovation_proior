# TIER: invalid
"""Emits an all-zero profile (sum == 0) -> infeasible -> must score 0."""
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    sys.stdout.write(" ".join(["0"] * n) + "\n")


if __name__ == "__main__":
    main()
