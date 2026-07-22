# TIER: invalid
"""Deliberately infeasible: every boat immediately tries to move left off the
left edge of the grid (col 0 - 1), which the checker must reject."""
import sys


def main():
    toks = sys.stdin.read().split()
    B = int(toks[1])
    out = [f"0 L" for _ in range(B)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
