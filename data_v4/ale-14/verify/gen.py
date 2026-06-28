#!/usr/bin/env python3
"""Instance generator for "Graph Coloring with Soft Conflicts" (ALE-Bench heuristic).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    n m k
    u_0 v_0 w_0
    u_1 v_1 w_1
    ...
    u_{m-1} v_{m-1} w_{m-1}

  * n  = number of vertices (0-indexed, 0..n-1).
  * m  = number of edges.
  * k  = number of available colors (each vertex gets a color in 0..k-1).
  * each edge is an undirected pair (u, v), u != v, with a positive integer
    weight w. There are no self-loops and no duplicate undirected pairs.

A coloring assigns each vertex one of k colors; an edge is a *conflict* if both
endpoints get the same color, and the cost of a coloring is the sum of the
weights of conflicting edges. The instances are deliberately built so that k
colors are NOT enough to avoid all conflicts (the chromatic number exceeds k),
so the optimum has strictly positive conflict weight -- this is a genuine
soft-conflict minimization, not a decision "is it k-colorable" problem.

Instance regime (the hard, conflict-forcing regime):
  * n in [400, 700].
  * k in [3, 6] (small, so colors are scarce relative to the graph density).
  * The graph is a mix of:
      - planted dense "cliquey" clusters (each forces > k same-cluster
        conflicts no matter how you color), plus
      - a random Erdos-Renyi background that couples clusters,
    with heavy-tailed edge weights so the optimizer must care *which* conflict
    edges it keeps, not just how many.
This is exactly the structure where a plain greedy coloring leaves a lot of
weight on the table and a good local search (tabu) can recover it.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0xC0107E5 ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(400, 700)
    k = rng.randint(3, 6)

    edges = {}  # (a,b) with a<b -> weight (keep last; dedup undirected)

    def add_edge(u, v, w):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edges:
            return
        edges[(a, b)] = w

    def weight():
        # heavy-tailed positive integer weights in [1, 1000]:
        # most edges are light, a few are very heavy, so the optimizer must
        # choose WHICH conflicts to keep, not merely how many.
        r = rng.random()
        if r < 0.80:
            return rng.randint(1, 20)
        elif r < 0.97:
            return rng.randint(20, 200)
        else:
            return rng.randint(200, 1000)

    # ---- planted dense clusters that force conflicts (each cluster size > k) ----
    # Clusters are drawn from the WHOLE vertex set (with overlap allowed across
    # clusters), so they densely couple the graph rather than partition it; the
    # overlap is what makes a globally-good coloring hard (a vertex shared by two
    # near-cliques is pulled toward different colors).
    num_clusters = rng.randint(12, 24)
    for _ in range(num_clusters):
        # cluster size strictly larger than k so it cannot be k-colored
        # without an internal monochromatic edge (pigeonhole on a clique core).
        size = rng.randint(k + 2, k + 10)
        members = rng.sample(range(n), size)
        # connect the cluster densely (each pair with high probability), so a
        # near-clique core guarantees leftover conflicts under k colors.
        p_in = rng.uniform(0.7, 1.0)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if rng.random() < p_in:
                    add_edge(members[i], members[j], weight())

    # ---- Erdos-Renyi background coupling everything ----
    # target a moderate average degree so the graph is substantive but not
    # trivially saturated; scaled to n.
    bg_edges = rng.randint(5 * n, 9 * n)
    for _ in range(bg_edges):
        u = rng.randrange(n)
        v = rng.randrange(n)
        add_edge(u, v, weight())

    # Guarantee the graph is non-trivial: at least n edges.
    while len(edges) < n:
        u = rng.randrange(n)
        v = rng.randrange(n)
        add_edge(u, v, weight())

    edge_list = list(edges.items())
    rng.shuffle(edge_list)
    m = len(edge_list)

    out = [f"{n} {m} {k}"]
    out.extend(f"{a} {b} {w}" for ((a, b), w) in edge_list)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
