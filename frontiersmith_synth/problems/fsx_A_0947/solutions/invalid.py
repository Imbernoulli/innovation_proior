# TIER: invalid
"""Deliberately infeasible: proposes a 'wall' between two cells that are not
4-adjacent. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it)); K = int(next(it))
    out = []
    out.append("1")
    out.append(f"1 0 0 {min(2, R - 1)} {min(2, C - 1)}")  # not 4-adjacent -> infeasible
    print("\n".join(out))


if __name__ == "__main__":
    main()
