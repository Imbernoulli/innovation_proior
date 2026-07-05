#!/usr/bin/env python3
# gen.py <testId>  -> prints ONE instance to stdout.
# Beehive apiary skin of a QAOA phase-separator routing problem.
#
# The apiary is a honeycomb comb: cells (physical qubits) laid out on a
# brick-wall / hexagonal lattice; two cells are ADJACENT iff they share a wall
# (a hardware edge -- the only cells between which a "waggle" two-body
# interaction can be performed directly).  Each bee (a logical qubit) starts in
# its home cell i (identity placement).  The nectar-sharing schedule is a list
# of required pairwise interactions (with multiplicities = how many waggles that
# pair must exchange); order is irrelevant because all waggles commute (they are
# diagonal RZZ phase rotations).  The comb only lets you interact ADJACENT cells,
# so bees must be walked around with SWAP moves.  Fewer SWAPs is better.
import sys, random


def build_comb(R, C):
    """Brick-wall honeycomb lattice: R rows x C cols of cells, degree<=3."""
    def nid(r, c):
        return r * C + c
    edges = []
    for r in range(R):
        for c in range(C - 1):
            edges.append((nid(r, c), nid(r, c + 1)))
    for r in range(R - 1):
        for c in range(C):
            if (r + c) % 2 == 0:
                edges.append((nid(r, c), nid(r + 1, c)))
    return R * C, edges


def main():
    t = int(sys.argv[1])
    rnd = random.Random(1000 + t)
    R = 3 + t
    C = 4 + t
    nq, edges = build_comb(R, C)

    # nectar-sharing (problem) graph: distinct unordered bee pairs
    ne_target = int(round(1.2 * nq))
    pairs = set()
    attempts = 0
    while len(pairs) < ne_target and attempts < ne_target * 40:
        a = rnd.randrange(nq)
        b = rnd.randrange(nq)
        if a != b:
            pairs.add((min(a, b), max(a, b)))
        attempts += 1
    pairs = sorted(pairs)

    maxw = 3
    flat = []
    for (u, v) in pairs:
        c = rnd.randint(1, maxw)
        for _ in range(c):
            flat.append((u, v))
    rnd.shuffle(flat)

    out = []
    out.append("%d %d" % (nq, len(edges)))
    for (a, b) in edges:
        out.append("%d %d" % (a, b))
    out.append("%d" % len(flat))
    for (u, v) in flat:
        out.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
