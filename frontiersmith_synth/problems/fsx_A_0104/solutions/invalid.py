# TIER: invalid
"""Emits an infeasible artifact: it stacks every pod at the same corner (0,0), so all
pods coincide and every triangle has area 0.  The checker must score this 0.0."""
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    out = ["0.0 0.0" for _ in range(n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
