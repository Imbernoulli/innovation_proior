#!/usr/bin/env python3
"""Generator for 'Phantom Ward Cartography' (fsx_A_1091).

Usage: python3 gen.py <testId>   (testId = 1..10), prints ONE instance to stdout.
All randomness is seeded by testId only -> bit-for-bit reproducible.

Instance: a town (graph) with planted true wards (communities), a handful of
'bridge' households with genuinely mixed ties, and a swap budget B.
Difficulty ladder: small/moderate separation -> large/adversarial strong
separation where unstructured (random or degree-targeted) edge noise barely
rotates the informative eigenvectors (Davis-Kahan), so only a coherent,
decoy-aligned perturbation fools the census pipeline there.
"""
import sys
import random


def build_case(seed, sizes, p_in, p_out, bridge_frac, budget_frac, cross_link=None):
    """Return (n, labels, edges, budget)."""
    rng = random.Random(seed)
    k = len(sizes)
    n = sum(sizes)
    labels = []
    for c, s in enumerate(sizes):
        labels += [c] * s
    # vertex ids are grouped by community (offset table)
    offs = [0]
    for s in sizes:
        offs.append(offs[-1] + s)

    n_bridge = max(2, int(round(bridge_frac * n)))
    # choose bridge vertices deterministically: spread across communities
    bridges = set()
    step = max(1, n // n_bridge)
    v = rng.randrange(step)
    while len(bridges) < n_bridge:
        bridges.add(v % n)
        v += step
    edges = set()
    for a in range(n):
        ca = labels[a]
        for b in range(a + 1, n):
            cb = labels[b]
            if a in bridges or b in bridges:
                # bridge household: mixed ties to everyone
                p = 0.5 * (p_in + 4.0 * p_out) if ca == cb else 0.5 * (p_in + 4.0 * p_out)
                if rng.random() < p:
                    edges.add((a, b))
            elif ca == cb:
                if rng.random() < p_in:
                    edges.add((a, b))
            else:
                p = p_out
                if cross_link is not None:
                    # cross_link: dict (c,d)->multiplier for specific community pairs
                    key = (min(ca, cb), max(ca, cb))
                    p = p_out * cross_link.get(key, 1.0)
                if rng.random() < p:
                    edges.add((a, b))
    # guarantee minimum degree 3 (deterministic repair pass)
    deg = [0] * n
    adj = [set() for _ in range(n)]
    for (a, b) in edges:
        deg[a] += 1
        deg[b] += 1
        adj[a].add(b)
        adj[b].add(a)
    for v in range(n):
        while deg[v] < 3:
            # attach to a random same-community vertex (deterministic rng stream)
            c = labels[v]
            lo, hi = offs[c], offs[c + 1]
            u = rng.randrange(lo, hi)
            if u == v or u in adj[v]:
                continue
            a, b = (v, u) if v < u else (u, v)
            edges.add((a, b))
            adj[v].add(u)
            adj[u].add(v)
            deg[v] += 1
            deg[u] += 1
    edges = sorted(edges)
    m = len(edges)
    budget = max(4, int(round(budget_frac * m)))
    return n, labels, edges, budget


def main():
    test_id = int(sys.argv[1])
    # difficulty ladder; cases 4,6,8,9,10 are the adversarial 'trap' cases:
    # strong community separation makes unstructured noise nearly useless.
    if test_id == 1:
        n, labels, edges, B = build_case(1101, [55, 55], 0.24, 0.055, 0.06, 0.10)
    elif test_id == 2:
        n, labels, edges, B = build_case(1102, [70, 70], 0.21, 0.045, 0.06, 0.10)
    elif test_id == 3:
        n, labels, edges, B = build_case(1103, [90, 90], 0.19, 0.038, 0.05, 0.10)
    elif test_id == 4:  # TRAP: strong separation, 2 wards
        n, labels, edges, B = build_case(1104, [110, 110], 0.17, 0.020, 0.07, 0.12)
    elif test_id == 5:
        n, labels, edges, B = build_case(1105, [80, 80, 80], 0.15, 0.028, 0.05, 0.10)
    elif test_id == 6:  # TRAP: strong separation, 3 wards
        n, labels, edges, B = build_case(1106, [100, 100, 100], 0.14, 0.011, 0.05, 0.12)
    elif test_id == 7:  # unequal wards, one small
        n, labels, edges, B = build_case(1107, [150, 110, 60], 0.13, 0.026, 0.05, 0.10)
    elif test_id == 8:  # TRAP: very strong separation, big budget
        n, labels, edges, B = build_case(1108, [200, 200], 0.115, 0.0075, 0.05, 0.14)
    elif test_id == 9:  # TRAP: strong separation, 3 wards, big budget
        n, labels, edges, B = build_case(1109, [140, 140, 140], 0.105, 0.009, 0.05, 0.14)
    elif test_id == 10:  # TRAP: 4 wards, strong separation, big budget
        n, labels, edges, B = build_case(1110, [120, 120, 120, 120], 0.11, 0.012, 0.05, 0.14)
    else:
        raise SystemExit("testId must be in 1..10")

    m = len(edges)
    out = []
    out.append("%d %d %d %d" % (n, m, len(set(labels)), B))
    out.append(" ".join(map(str, labels)))
    for (a, b) in edges:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
