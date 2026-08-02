# TIER: invalid
"""Deliberately infeasible: dumps way more solute than MAXX ever allows."""
import sys


def main():
    toks = sys.stdin.read().split()
    K = int(toks[0])
    print(' '.join(['999999999'] * K))


if __name__ == "__main__":
    main()
