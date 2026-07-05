# TIER: trivial
"""Emit the triangle profile -- exactly the checker's internal baseline.
Scores ~0.1 by construction."""
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    f = [min(i + 1, n - i) for i in range(n)]
    sys.stdout.write(" ".join(map(str, f)) + "\n")


if __name__ == "__main__":
    main()
