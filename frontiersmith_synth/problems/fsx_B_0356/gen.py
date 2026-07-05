#!/usr/bin/env python3
"""gen.py <testId> -> ONE instance of the cold-chain qubit-routing problem.

Skin: a "vaccine cold-chain" co-inspection scheduler. The physical qubits are
refrigerated slots of a heavy-hex-inspired storage lattice (the backend coupling
map); the required two-qubit interactions are the ZZ terms of a QAOA cost circuit
(pairs of lots that must be jointly cross-checked while their slots are adjacent).
The solver must emit a routing (SWAP schedule) that realises every ZZ term, and is
scored on the number of SWAP moves used (cold-chain transfers to be minimised).

Difficulty ladder: testId 1..N grows the lattice and the number of QAOA layers.
All randomness is seeded by testId ONLY -> bit-for-bit reproducible.
"""
import sys, random


def build():
    t = int(sys.argv[1])
    rng = random.Random(999_983 * t + 12_347)

    # --- heavy-hex-inspired storage lattice: a full grid coupling map ---
    rows = 3 + (t // 4)
    cols = 4 + (t // 4)
    Q = rows * cols

    def nid(r, c):
        return r * cols + c

    edges = []
    eset = set()

    def add_edge(u, v):
        k = (u, v) if u < v else (v, u)
        if k not in eset and u != v:
            eset.add(k)
            edges.append(k)

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                add_edge(nid(r, c), nid(r, c + 1))
            if r + 1 < rows:
                add_edge(nid(r, c), nid(r + 1, c))

    def rc(x):
        return (x // cols, x % cols)

    def manh(u, v):
        (r1, c1), (r2, c2) = rc(u), rc(v)
        return abs(r1 - r2) + abs(c1 - c2)

    # --- QAOA cost graph on the logical lots (identity initial placement) ---
    M = max(Q, round(1.3 * Q))
    prob = set()
    tries = 0
    while len(prob) < M and tries < 200 * M:
        tries += 1
        a = rng.randrange(Q)
        b = rng.randrange(Q)
        if a == b:
            continue
        d = manh(a, b)
        # bias hard: prefer NON-adjacent lots so routing is genuinely required
        if d == 1 and rng.random() < 0.75:
            continue
        prob.add((a, b) if a < b else (b, a))
    prob = sorted(prob)

    layers = 1 if t <= 5 else 2
    inter = []
    for _ in range(layers):
        inter.extend(prob)
    rng.shuffle(inter)

    out = []
    out.append("%d %d %d" % (Q, len(edges), len(inter)))
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    for (a, b) in inter:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    build()
