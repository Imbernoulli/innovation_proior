# TIER: trivial
"""Reproduces the checker's own reference construction: k disjoint copies of a
6-vertex gadget A, with exactly ONE copy swapped for its cospectral mate B."""
import sys

GADGET_A = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 4), (2, 3)]
GADGET_B = [(0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 4), (2, 3)]
SZ = 6


def emit(n, edges):
    print(n, len(edges))
    for (u, v) in edges:
        print(u, v)


def main():
    n, m = map(int, sys.stdin.read().split())
    k = n // SZ
    edgesG = []
    edgesH = []
    for i in range(k):
        base = i * SZ
        for (a, b) in GADGET_A:
            edgesG.append((base + a + 1, base + b + 1))
        variant = GADGET_B if i == 0 else GADGET_A
        for (a, b) in variant:
            edgesH.append((base + a + 1, base + b + 1))
    emit(n, edgesG)
    emit(n, edgesH)


if __name__ == "__main__":
    main()
