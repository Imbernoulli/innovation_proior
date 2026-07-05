# TIER: trivial
"""Bubble-sort comparator network: n*(n-1)/2 comparators. This exactly reproduces
the checker's internal baseline B, so it scores ~0.1 on every test."""
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    out = []
    for i in range(n - 1):
        for j in range(n - 1 - i):
            out.append("%d %d" % (j, j + 1))   # min -> lower wire
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
