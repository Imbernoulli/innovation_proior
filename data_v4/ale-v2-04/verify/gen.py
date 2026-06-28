#!/usr/bin/env python3
"""Instance generator for "Graph Coloring (Minimize Colors)" (ALE-Bench heuristic).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    n m
    u_0 v_0
    u_1 v_1
    ...
    u_{m-1} v_{m-1}

  * n  = number of vertices (0-indexed, 0..n-1).
  * m  = number of edges.
  * each edge is an undirected pair (u, v), u != v. There are no self-loops and
    no duplicate undirected pairs. The graph is simple and undirected.

The task is to PROPERLY color the graph -- assign each vertex a color so that no
edge is monochromatic -- using AS FEW COLORS AS POSSIBLE. The minimum number of
colors is the chromatic number chi(G); computing it is NP-hard, and the benchmark
scores a coloring by how few colors it uses (an improper coloring scores 0).

How the instances are built (so a real gap exists between a naive greedy and the
best achievable, while keeping a known upper bound on chi):

  * We first PLANT a hidden proper C-coloring: partition the n vertices into C
    color classes of (roughly) equal size. We then add edges ONLY between
    different classes -- never inside a class. By construction the planted
    partition is a proper C-coloring, so chi(G) <= C. C is chosen modestly small
    relative to n, so the graph is genuinely few-colorable.

  * Inter-class edges are added DENSELY (a high probability between every pair of
    classes), which (a) makes the C classes near-complete-multipartite so chi is
    very close to C from below as well (the planted classes behave like a clique
    on the class level), and (b) raises the average degree so a plain first-fit
    greedy -- the scorer's baseline -- opens SEVERAL more colors than C. This is
    the classic regime where first-fit / largest-first greedy wastes colors and a
    DSATUR + tabu local search recovers them: greedy's degree ordering keeps
    "discovering" that the next vertex conflicts with all currently open colors
    and opens a fresh one, whereas DSATUR's saturation ordering and tabu descent
    pack the coloring back down toward C.

  * To make sure greedy is fooled (and the planted partition is not trivially
    recoverable), we also lightly perturb: a few extra random inter-class edges,
    and the class assignment is random, so the optimizer cannot just read the
    structure off the vertex indices.

Regime: n in [350, 550]; C in [a moderate band], inter-class density high. The
result: a known upper bound chi <= C, a near-complete-multipartite core pinning
chi close to C, and a first-fit greedy that uses clearly more than C colors --
exactly the gap a strong color-minimization heuristic is meant to close.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0xC010F1 ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(350, 550)
    # number of planted color classes (the planted-coloring upper bound on chi).
    C = rng.randint(8, 14)

    # random balanced partition into C classes.
    cls = [i % C for i in range(n)]
    rng.shuffle(cls)
    classes = [[] for _ in range(C)]
    for v in range(n):
        classes[cls[v]].append(v)

    edges = set()

    def add_edge(u, v):
        if u == v or cls[u] == cls[v]:
            return  # never add an intra-class edge: keeps planted coloring proper
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))

    # Dense inter-class edges: between every ordered pair of distinct classes,
    # connect each cross pair with high probability p. This makes the C classes a
    # near-complete C-partite graph (chi pinned near C) and a high-degree graph
    # that fools first-fit greedy into opening extra colors.
    p = rng.uniform(0.55, 0.78)
    for a in range(C):
        for b in range(a + 1, C):
            for u in classes[a]:
                for w in classes[b]:
                    if rng.random() < p:
                        add_edge(u, w)

    # The above can be very dense; thin it down to a target average degree so the
    # instance stays a heuristic problem (not a trivially saturated clique) while
    # remaining hard for greedy. We KEEP a random subset of the inter-class edges.
    edge_list = list(edges)
    rng.shuffle(edge_list)
    # target total edges ~ a multiple of n, capped by what we generated.
    target = rng.randint(9 * n, 14 * n)
    if target < len(edge_list):
        edge_list = edge_list[:target]

    # Guarantee the graph is non-trivial: at least n edges. (Add more inter-class
    # edges if we somehow fell short.)
    kept = set(edge_list)
    while len(kept) < n:
        u = rng.randrange(n)
        w = rng.randrange(n)
        if cls[u] == cls[w]:
            continue
        a, b = (u, w) if u < w else (w, u)
        kept.add((a, b))
    edge_list = list(kept)
    rng.shuffle(edge_list)

    m = len(edge_list)
    out = [f"{n} {m}"]
    out.extend(f"{a} {b}" for (a, b) in edge_list)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
