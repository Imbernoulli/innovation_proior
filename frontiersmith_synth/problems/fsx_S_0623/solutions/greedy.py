# TIER: greedy
"""The obvious 'textbook switch' approach: find ONE cospectral-mate gadget, and
apply it a small, fixed number of times (here: 2), regardless of instance size.
This is a natural first attempt -- more than the bare minimum single switch -- but
it does not scale with n, so on large instances it leaves almost all of the
available divergence on the table."""
import sys

GADGET_A = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 4), (2, 3)]
GADGET_B = [(0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 4), (2, 3)]
SZ = 6
FIXED_SWAPS = 2


def emit(n, edges):
    print(n, len(edges))
    for (u, v) in edges:
        print(u, v)


def main():
    n, m = map(int, sys.stdin.read().split())
    k = n // SZ
    swaps = min(FIXED_SWAPS, k)
    edgesG = []
    edgesH = []
    for i in range(k):
        base = i * SZ
        for (a, b) in GADGET_A:
            edgesG.append((base + a + 1, base + b + 1))
        variant = GADGET_B if i < swaps else GADGET_A
        for (a, b) in variant:
            edgesH.append((base + a + 1, base + b + 1))
    emit(n, edgesG)
    emit(n, edgesH)


if __name__ == "__main__":
    main()
