# TIER: invalid
"""Deliberately infeasible artifact: install every candidate mutation, blowing
straight through the mutation budget K (n > K always by construction)."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    print(n)
    print(*range(n))


if __name__ == "__main__":
    main()
