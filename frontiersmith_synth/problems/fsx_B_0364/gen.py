#!/usr/bin/env python3
# gen.py <testId>  -- prints ONE instance of the deep-sea cable-swap routing problem.
# testId 1..10 = difficulty ladder (small mesh/few ops -> large mesh/many ops).
# All randomness seeded ONLY by testId (deterministic).
#
# Deep-sea cable network:
#   - `n` junction boxes on the seabed (physical sites 0..n-1).
#   - Cables connect them; every box currently holds exactly one signal channel (a token).
#   - A schedule of `q` splice operations, each naming two channels (tokens) that must be
#     co-located on directly-cabled boxes to be spliced, executed IN ORDER.
#   - A splice can only run when its two channels sit on adjacent (cabled) boxes.
#   - To make distant channels adjacent, the crew performs SWAPs across a single cable
#     (exchange the channels on the two endpoint boxes).  Minimise total SWAPs.
#
# The graph is a rectangular seabed mesh (rows x cols) plus a few long-haul trunk cables.

import sys
import random


def build_instance(test_id):
    rng = random.Random(70000 + 101 * test_id)

    # difficulty ladder
    rows = 3 + (test_id // 2)          # 3..8
    cols = 3 + test_id                 # 4..13
    n = rows * cols
    q = 8 + 16 * test_id               # 24..168

    def sid(r, c):
        return r * cols + c

    edges = set()
    # rectangular mesh cables
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.add((sid(r, c), sid(r, c + 1)))
            if r + 1 < rows:
                edges.add((sid(r, c), sid(r + 1, c)))

    # a few long-haul trunk cables (variant flavour) -- keep the graph sparse
    n_trunks = 1 + test_id // 3
    added = 0
    attempts = 0
    while added < n_trunks and attempts < 200:
        attempts += 1
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edges:
            continue
        edges.add((a, b))
        added += 1

    edge_list = sorted(edges)

    # initial placement: a permutation of tokens 0..n-1 onto sites 0..n-1.
    placement = list(range(n))
    rng.shuffle(placement)   # placement[site] = token on that site

    # build adjacency for distance-aware op generation
    adj = [[] for _ in range(n)]
    for (a, b) in edge_list:
        adj[a].append(b)
        adj[b].append(a)

    site_of = [0] * n
    for s, tok in enumerate(placement):
        site_of[tok] = s

    def bfs_dist(src):
        dist = [-1] * n
        dist[src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for x in frontier:
                for y in adj[x]:
                    if dist[y] < 0:
                        dist[y] = dist[x] + 1
                        nxt.append(y)
            frontier = nxt
        return dist

    # splice schedule with LOCALITY: with prob p reuse a recently-touched token,
    # otherwise draw a fresh token.  Locality rewards good positioning strategies.
    ops = []
    recent = []
    guaranteed_far = 0
    for _ in range(q):
        if recent and rng.random() < 0.55:
            a = rng.choice(recent[-6:])
        else:
            a = rng.randrange(n)
        b = rng.randrange(n)
        while b == a:
            b = rng.randrange(n)
        ops.append((a, b))
        recent.append(a)
        recent.append(b)
        # count how many ops are non-adjacent in the HOME placement
        da = bfs_dist(site_of[a])
        if da[site_of[b]] > 1:
            guaranteed_far += 1

    # ensure the instance is non-degenerate (some ops need real routing).
    # (In practice a mesh + random pairs makes almost all pairs distance>1.)
    if guaranteed_far == 0:
        # force one far op
        far_a = 0
        db = bfs_dist(site_of[far_a])
        far_site = max(range(n), key=lambda s: db[s])
        far_b = placement[far_site]
        if far_b == far_a:
            far_b = placement[(far_site + 1) % n]
        ops[0] = (far_a, far_b)

    return n, edge_list, placement, ops


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    n, edge_list, placement, ops = build_instance(test_id)

    out = []
    out.append("%d %d %d" % (n, len(edge_list), len(ops)))
    for (a, b) in edge_list:
        out.append("%d %d" % (a, b))
    out.append(" ".join(str(x) for x in placement))
    for (a, b) in ops:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
