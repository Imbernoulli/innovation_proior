# TIER: trivial
"""Reproduces the checker's own baseline: session i loads references
i, i+1, ..., i+k-1 (mod n), all on the left pan. Always feasible, badly
correlated (adjacent sessions share almost all their loaded references)."""
import sys


def main():
    n, k = map(int, sys.stdin.read().split()[:2])
    out = []
    for i in range(n):
        row = [0] * n
        for t in range(k):
            row[(i + t) % n] = 1
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
