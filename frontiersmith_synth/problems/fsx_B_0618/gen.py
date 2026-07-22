#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of the gemstone-sticker unfolding problem.

The instance is built deterministically from testId alone (1..10, difficulty ladder).
The "gemstone" is a chain of P rectangular facet-panels (each H rows x L columns of
unit-square facets). Within a panel all facets are glued FLAT to their row/column
neighbours (a rigid coplanar sheet). Between panel i and panel i+1 there are L
CANDIDATE fusion edges (one per column c=0..L-1); each is a 90-degree FOLD, and
because the panels are otherwise fully connected internally, a feasible hinge set
may use AT MOST ONE of the L candidates per boundary (any second one would close a
cycle). Which column is chosen determines where the folded panel lands in the plane.
"""
import sys

DIRS = [0, 1, 2, 3]  # N,E,S,W


def nid(base, r, c, L):
    return base + r * L + c


def build_edges(P, H, L, signs):
    """Return (n, edges) where edges is a list of (u, su, v, sv)."""
    n = P * H * L
    bases = [i * H * L for i in range(P)]
    edges = []
    # 1) all internal (flat) edges of every panel, panel by panel
    for i in range(P):
        base = bases[i]
        for r in range(H - 1):
            for c in range(L):
                edges.append((nid(base, r, c, L), 2, nid(base, r + 1, c, L), 0))
        for c in range(L - 1):
            edges.append((nid(base, H - 1, c, L), 1, nid(base, H - 1, c + 1, L), 3))
    # 2) all boundary (fold) candidate edges, boundary by boundary, column order
    for i in range(P - 1):
        s = signs[i]
        baseA = bases[i]
        baseB = bases[i + 1]
        for c in range(L):
            A = nid(baseA, H - 1, c, L)
            B = nid(baseB, 0, c, L)
            if s == 1:
                edges.append((A, 2, B, 3))
            else:
                edges.append((A, 2, B, 1))
    return n, edges


# testId -> (P, H, L, signs)   (signs has length P-1, each +1 or -1)
CASES = {
    1: (2, 2, 2, [1]),
    2: (3, 2, 2, [1, 1]),
    3: (3, 2, 3, [1, 1]),
    4: (4, 2, 2, [1, 1, 1]),
    5: (4, 3, 3, [1, 1, 1]),
    6: (4, 2, 3, [1, 1, 1]),
    7: (5, 2, 2, [1, 1, 1, 1]),
    8: (6, 2, 3, [1, 1, 1, 1, 1]),
    9: (6, 3, 3, [1, 1, 1, 1, 1]),
    10: (10, 2, 3, [1, 1, 1, 1, 1, 1, 1, 1, 1]),
}


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    testId = int(sys.argv[1])
    testId = ((testId - 1) % 10) + 1
    P, H, L, signs = CASES[testId]
    n, edges = build_edges(P, H, L, signs)
    penalty = max(4, L)
    out = [f"{n} {len(edges)} {penalty}"]
    for (u, su, v, sv) in edges:
        out.append(f"{u} {su} {v} {sv}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
