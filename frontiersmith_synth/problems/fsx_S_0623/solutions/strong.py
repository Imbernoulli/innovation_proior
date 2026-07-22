# TIER: strong
"""The insight: eigenvalues of a disjoint union are just the multiset union of the
pieces' eigenvalues. Since gadget A and gadget B are exactly cospectral, swapping
ANY subset of the k blocks from A to B leaves the WHOLE graph's spectrum untouched
-- there is zero marginal cost to swapping more blocks. So instead of one (or a
few) localized switches, compose the swap across EVERY block: replace all k copies
of A with B. Divergence (both the summed-diameter term and the degree-sequence
term) accumulates roughly linearly in k, while cospectrality is preserved for
free, all the way up."""
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
        for (a, b) in GADGET_B:
            edgesH.append((base + a + 1, base + b + 1))
    emit(n, edgesG)
    emit(n, edgesH)


if __name__ == "__main__":
    main()
