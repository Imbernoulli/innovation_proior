# TIER: greedy
"""Binary-orthant heuristic: use every vector in {0,1}^n.

Any three distinct 0/1 vectors cannot sum to 0 mod 3 in every coordinate (that
would force all three to agree in each coordinate, i.e. be identical), so
{0,1}^n is always a valid cap set. Size = 2^n, well above the diagonal
baseline but far from optimal."""
import sys
import itertools


def main():
    n = int(sys.stdin.readline().split()[0])
    lines = []
    for v in itertools.product((0, 1), repeat=n):
        lines.append(" ".join(map(str, v)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
