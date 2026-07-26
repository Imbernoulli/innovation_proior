#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE heat-trace-equalizer instance to stdout.
Deterministic: all randomness is seeded from testId only (no wall time, no OS entropy).

Topology: K "radiator clusters", each an internally well-connected (cycle + chords)
subgraph, joined in a path by exactly one bridge edge between consecutive clusters.
The bridge edges are true graph bridges (their removal disconnects the graph) but are
NOT labeled as such in the output -- the solver must find them itself if it wants to
exploit the structural split.

Targets are the heat-kernel trace Tr(exp(-t L_w)) of a HIDDEN continuous weight
profile w*, evaluated at a grid of times spanning several decades, with a small
deterministic per-time measurement jitter added (so no integer weight vector can
reach zero error -- this keeps headroom for the score).
"""
import sys
import random
import numpy as np


def build_cluster(nodes, rng, cap_lo, cap_hi):
    """Cycle + extra chords on `nodes` -> 2-edge-connected (no single edge is a bridge)."""
    m = len(nodes)
    edges = []
    if m == 1:
        return edges
    for i in range(m):
        u, v = nodes[i], nodes[(i + 1) % m]
        edges.append([u, v, rng.randint(cap_lo, cap_hi)])
    extra = max(0, m - 2)
    pairs = set()
    tries = 0
    while len(pairs) < extra and tries < 10 * extra + 60:
        tries += 1
        i, j = rng.randrange(m), rng.randrange(m)
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        if (a, b) in pairs:
            continue
        if (b - a) == 1 or (a == 0 and b == m - 1):
            continue
        pairs.add((a, b))
        edges.append([nodes[a], nodes[b], rng.randint(cap_lo, cap_hi)])
    return edges


def gen_instance(test_id):
    rng = random.Random(20260726 + 977 * test_id)

    if test_id <= 3:
        K, base_m = 2, 4 + (test_id - 1)
    elif test_id <= 6:
        K, base_m = 3, 4 + (test_id - 4)
    elif test_id <= 8:
        K, base_m = 3, 7 + (test_id - 7)
    else:
        K, base_m = 4, 6 + (test_id - 9)

    cluster_sizes = [base_m + rng.randint(0, 1) for _ in range(K)]
    clusters, start = [], 0
    for sz in cluster_sizes:
        clusters.append(list(range(start, start + sz)))
        start += sz
    n = start

    cap_lo, cap_hi = 3, 10
    edges = []
    for cl in clusters:
        edges.extend(build_cluster(cl, rng, cap_lo, cap_hi))
    for k in range(K - 1):
        u, v = clusters[k][-1], clusters[k + 1][0]
        edges.append([u, v, rng.randint(cap_lo, cap_hi)])
    m = len(edges)
    bridge_idx = set(range(m - (K - 1), m))

    if test_id % 2 == 1:
        a_int = rng.uniform(0.60, 0.85)
        a_cut = rng.uniform(0.10, 0.30)
    else:
        a_int = rng.uniform(0.30, 0.48)
        a_cut = rng.uniform(0.65, 0.90)

    wstar = []
    for idx, (u, v, cap) in enumerate(edges):
        jitter = 1.0 + rng.uniform(-0.12, 0.12)
        alpha = a_cut if idx in bridge_idx else a_int
        wstar.append(max(0.4, min(cap, alpha * cap * jitter)))

    T = 9
    t0, r = 0.015, 2.6
    ts = [t0 * (r ** j) for j in range(T)]
    noise = [rng.uniform(-0.035, 0.035) for _ in range(T)]

    L = np.zeros((n, n))
    for (u, v, cap), w in zip(edges, wstar):
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    eigs = np.linalg.eigvalsh(L)
    targets = [float(np.sum(np.exp(-t * eigs))) * (1.0 + nz) for t, nz in zip(ts, noise)]

    return n, edges, ts, targets


def main():
    test_id = int(sys.argv[1])
    n, edges, ts, targets = gen_instance(test_id)
    m, T = len(edges), len(ts)
    out = [f"{n} {m} {T}"]
    for u, v, cap in edges:
        out.append(f"{u} {v} {cap}")
    out.append(" ".join(f"{t:.10g}" for t in ts))
    out.append(" ".join(f"{g:.10g}" for g in targets))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
