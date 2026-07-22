# TIER: invalid
"""Emits an infeasible demolition order: pier ids shifted down by one, so pier n
never appears and pier id 0 (out of the required 1..n range) appears instead.
Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    print(" ".join(str(i) for i in range(0, n)))


if __name__ == "__main__":
    main()
