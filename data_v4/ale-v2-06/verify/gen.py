#!/usr/bin/env python3
"""Instance generator for "Dense Weighted Independent Set" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    n m
    w_0 w_1 ... w_{n-1}
    a_1 b_1
    a_2 b_2
    ...
    a_m b_m

where:
  * n is the number of vertices (chosen deterministically from the seed in [600, 1200]),
  * m is the number of undirected edges,
  * w_i is the integer weight of vertex i (1 <= w_i <= 1000),
  * each of the m following lines is an undirected edge {a, b}, 0 <= a, b < n, a != b.
    No edge is repeated and there are no self-loops.

Design intent: a *dense* weighted-MWIS instance where the obvious "greedy by weight"
or "greedy by weight/degree" gives a clearly suboptimal independent set, so a real
local search (the tightness-based (1,2)-swap / plateau search) has room to win. We
build the graph in the complement-friendly dense regime: average degree is a large
fraction of n, so any independent set is small and which few vertices you keep is a
genuinely hard combinatorial choice. To make the weight signal non-trivial (so that
"just take the heaviest" is a trap), we plant a few moderately-weighted *independent*
structures that the greedy overlooks because each individual vertex is lighter than a
heavy-but-highly-connected vertex that blocks many others.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x1E5C0DE ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(600, 1200)

    # Edge density: dense regime. Average degree is a large fraction of n so that the
    # maximum independent set is small and selecting it is hard. We pick a target
    # edge probability p in a dense band, then perturb per-vertex connectivity.
    p = rng.uniform(0.30, 0.55)

    # Per-vertex "popularity" multiplier to break the uniform-random structure: some
    # vertices are hubs (connect to many), some are sparse. A hub with a big weight is
    # exactly the greedy trap -- taking it blocks a large independent neighbourhood.
    pop = [rng.uniform(0.4, 1.6) for _ in range(n)]

    edges = set()

    # 1) Erdos-Renyi-ish backbone with per-vertex popularity scaling.
    #    To keep generation O(n^2) bounded for n<=1200 (~1.4M pairs) we sample directly.
    for i in range(n):
        pi = pop[i]
        for j in range(i + 1, n):
            pij = p * pi * pop[j]
            if pij > 1.0:
                pij = 1.0
            if rng.random() < pij:
                edges.add((i, j))

    # 2) Plant a few sparse independent "pockets": pick a random subset of vertices,
    #    declare them mutually non-adjacent (remove any backbone edges among them), and
    #    give them moderate weights. The greedy-by-weight overlooks these pockets because
    #    each pocket vertex is individually lighter than the heavy hubs; only by taking
    #    the WHOLE pocket (which a local search discovers) do they pay off. This is the
    #    structure that separates greedy from local search.
    num_pockets = rng.randint(2, 5)
    pocket_vertices = set()
    for _ in range(num_pockets):
        size = rng.randint(8, 18)
        verts = rng.sample(range(n), min(size, n))
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                u, v = verts[a], verts[b]
                if u > v:
                    u, v = v, u
                edges.discard((u, v))  # make them mutually independent
        pocket_vertices.update(verts)

    # Weights: base random weight; hubs (high popularity) get a weight BONUS so that
    # "take the heaviest first" is genuinely tempting and genuinely wrong. Pocket
    # vertices get a moderate weight (good in aggregate, mediocre individually).
    w = [0] * n
    for i in range(n):
        base = rng.randint(50, 700)
        # hub bonus correlated with popularity -> heavy-but-blocking vertices
        bonus = int((pop[i] - 0.4) / 1.2 * rng.randint(0, 400))
        w[i] = base + bonus
        if w[i] > 1000:
            w[i] = 1000
        if w[i] < 1:
            w[i] = 1
    for v in pocket_vertices:
        # moderate, not the absolute heaviest
        w[v] = rng.randint(200, 550)

    edge_list = sorted(edges)
    m = len(edge_list)

    out = [f"{n} {m}", " ".join(str(x) for x in w)]
    out.extend(f"{a} {b}" for (a, b) in edge_list)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
