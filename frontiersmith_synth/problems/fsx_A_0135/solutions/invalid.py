# TIER: invalid
"""Emit an infeasible density (all zeros -> total population != S). Scores 0."""
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    sys.stdout.write(" ".join("0" for _ in range(n)) + "\n")


if __name__ == "__main__":
    main()
