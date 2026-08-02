#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE shard-key-assignment instance to stdout.
Deterministic: seeded only by testId. Difficulty ladder small -> large/adversarial.

Each instance describes N keys (with sizes/weights), a previous-epoch shard
assignment (some keys may be brand new, prev=-1), and a weighted transaction
trace linking pairs of keys that are read/written together. A solver must
output a NEW shard assignment.

Plants (>=3, in fact >=6) "transaction-clustered" cases where keys are
organised into dense co-transacting cliques: any assignment that only
optimizes for size-uniformity (hash/round-robin, or balanced bin-packing)
scatters clique members across shards and pays enormous cross-shard
transaction cost, while co-locating each clique (accepting size imbalance)
is far cheaper. Several cases also carry a *good* previous assignment with a
heavy migration-cost coefficient, so ignoring resharding stability (blindly
recomputing from scratch) is separately punished.
"""
import sys
import random


def make_weights(rng, n, lo=1, hi=20):
    return [rng.randint(lo, hi) for _ in range(n)]


def add_edge(edges, u, v, c):
    if u == v:
        return
    key = (u, v) if u < v else (v, u)
    edges[key] = edges.get(key, 0) + c


def clique_edges(rng, edges, members, wlo, whi, density=1.0):
    """Dense (or near-dense) co-transaction structure inside one cluster."""
    m = len(members)
    for i in range(m):
        for j in range(i + 1, m):
            if rng.random() <= density:
                c = rng.randint(wlo, whi)
                add_edge(edges, members[i], members[j], c)


def chain_noise(rng, edges, n, count, wlo, whi):
    """A handful of weak, unstructured pairwise transactions (no planted
    cluster) -- present in warm-up cases so they are not transaction-free."""
    for _ in range(count):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        c = rng.randint(wlo, whi)
        add_edge(edges, u, v, c)


def ring_bridge_edges(rng, edges, clusters, wlo, whi, per_link=1):
    """Connect consecutive clusters in a ring with a handful of moderate-
    weight inter-cluster edges. This guarantees NO partition can drive
    cross-shard cost to zero just by giving every cluster its own shard (or
    by merging clusters freely) -- some ring edge is always cut unless
    every cluster collapses onto the same shard, which wrecks the skew
    term instead. Keeps the strong/local-search reference from saturating
    the score by finding a truly zero-cost assignment."""
    m = len(clusters)
    if m < 2:
        return
    for ci in range(m):
        a = clusters[ci]
        b = clusters[(ci + 1) % m]
        for _ in range(per_link):
            u = rng.choice(a)
            v = rng.choice(b)
            c = rng.randint(wlo, whi)
            add_edge(edges, u, v, c)


def partition_into_clusters(rng, n, sizes):
    """Split key ids 0..n-1 into clusters of the given sizes (any leftover
    keys form singleton/no-cluster keys). Order of ids inside each cluster
    is shuffled so weight-sort order does NOT correlate with cluster id."""
    ids = list(range(n))
    rng.shuffle(ids)
    clusters = []
    pos = 0
    for s in sizes:
        clusters.append(ids[pos:pos + s])
        pos += s
    leftover = ids[pos:]
    return clusters, leftover


def good_prev_assignment(rng, n, clusters, K, keep_frac=1.0):
    """A previous-epoch assignment that already co-locates most clusters
    (one cluster -> one shard, round-robin over shards), with a fraction of
    keys marked as brand-new (-1, no migration cost either way) and a small
    number of keys deliberately mis-placed to simulate epoch drift."""
    prev = [-1] * n
    for ci, cluster in enumerate(clusters):
        shard = ci % K
        for k in cluster:
            prev[k] = shard
    # mark some keys as brand new (no prior assignment)
    for k in range(n):
        if prev[k] != -1 and rng.random() > keep_frac:
            prev[k] = -1
    return prev


def build(tid):
    rng = random.Random(30000 + 131 * tid)

    if tid == 1:
        n, K = 6, 2
        weights = make_weights(rng, n, 3, 12)
        edges = {}
        clusters = []
        prev = [-1] * n
        A, B, G = 1, 5, 5
    elif tid == 2:
        n, K = 10, 3
        weights = make_weights(rng, n, 2, 15)
        edges = {}
        chain_noise(rng, edges, n, 4, 1, 4)
        clusters = []
        prev = [-1] * n
        A, B, G = 1, 3, 3
    elif tid == 3:
        n, K = 12, 3
        weights = make_weights(rng, n, 1, 20)
        clusters, leftover = partition_into_clusters(rng, n, [4, 4])
        edges = {}
        for cl in clusters:
            clique_edges(rng, edges, cl, 6, 14, density=0.85)
        chain_noise(rng, edges, n, 2, 1, 3)
        ring_bridge_edges(rng, edges, clusters, 5, 10)
        prev = [-1] * n
        A, B, G = 1, 8, 0
    elif tid == 4:
        n, K = 16, 4
        weights = make_weights(rng, n, 1, 20)
        clusters, leftover = partition_into_clusters(rng, n, [4, 4, 4, 4])
        edges = {}
        for cl in clusters:
            clique_edges(rng, edges, cl, 15, 25, density=1.0)
        ring_bridge_edges(rng, edges, clusters, 10, 20, per_link=3)
        prev = [-1] * n
        A, B, G = 1, 15, 0
    elif tid == 5:
        n, K = 24, 4
        weights = make_weights(rng, n, 1, 20)
        clusters, leftover = partition_into_clusters(rng, n, [4, 4, 4, 4, 4, 4])
        edges = {}
        for cl in clusters:
            clique_edges(rng, edges, cl, 15, 25, density=0.9)
        chain_noise(rng, edges, n, 3, 1, 3)
        ring_bridge_edges(rng, edges, clusters, 6, 14, per_link=2)
        prev = [-1] * n
        A, B, G = 1, 20, 0
    elif tid == 6:
        n, K = 14, 3
        weights = make_weights(rng, n, 2, 18)
        clusters, leftover = partition_into_clusters(rng, n, [4, 4, 3])
        edges = {}
        for cl in clusters:
            clique_edges(rng, edges, cl, 4, 10, density=0.8)
        ring_bridge_edges(rng, edges, clusters, 4, 9)
        prev = good_prev_assignment(rng, n, clusters, K, keep_frac=0.85)
        A, B, G = 1, 5, 25
    elif tid == 7:
        n, K = 20, 4
        weights = make_weights(rng, n, 1, 20)
        clusters, leftover = partition_into_clusters(rng, n, [4, 4, 4, 4])
        edges = {}
        for cl in clusters:
            clique_edges(rng, edges, cl, 10, 20, density=0.9)
        chain_noise(rng, edges, n, 4, 1, 3)
        ring_bridge_edges(rng, edges, clusters, 6, 12, per_link=2)
        prev = good_prev_assignment(rng, n, clusters, K, keep_frac=0.7)
        A, B, G = 1, 12, 15
    elif tid == 8:
        n, K = 28, 5
        weights = make_weights(rng, n, 1, 20)
        clusters, leftover = partition_into_clusters(rng, n, [4, 4, 4, 4, 4, 4, 4])
        edges = {}
        for cl in clusters:
            clique_edges(rng, edges, cl, 10, 22, density=0.85)
        chain_noise(rng, edges, n, 4, 1, 3)
        ring_bridge_edges(rng, edges, clusters, 7, 14, per_link=2)
        prev = good_prev_assignment(rng, n, clusters, K, keep_frac=0.6)
        A, B, G = 1, 15, 18
    elif tid == 9:
        n, K = 30, 5
        weights = make_weights(rng, n, 1, 25)
        clusters, leftover = partition_into_clusters(rng, n, [5, 5, 5, 5, 5])
        edges = {}
        for cl in clusters:
            clique_edges(rng, edges, cl, 10, 24, density=0.85)
        chain_noise(rng, edges, n, 5, 1, 4)
        ring_bridge_edges(rng, edges, clusters, 7, 15, per_link=2)
        # partly-good, partly-drifted previous assignment (some clusters
        # already broken up by an earlier resharding event)
        prev = good_prev_assignment(rng, n, clusters, K, keep_frac=0.55)
        for k in leftover:
            if rng.random() < 0.5:
                prev[k] = rng.randrange(K)
        A, B, G = 1, 18, 20
    else:  # tid == 10 (largest / adversarial)
        n, K = 36, 6
        weights = make_weights(rng, n, 1, 25)
        clusters, leftover = partition_into_clusters(
            rng, n, [4, 5, 4, 5, 4, 4, 5, 4])
        edges = {}
        for cl in clusters:
            clique_edges(rng, edges, cl, 10, 25, density=0.85)
        chain_noise(rng, edges, n, 6, 1, 4)
        ring_bridge_edges(rng, edges, clusters, 8, 16, per_link=2)
        prev = good_prev_assignment(rng, n, clusters, K, keep_frac=0.5)
        for k in leftover:
            if rng.random() < 0.5:
                prev[k] = rng.randrange(K)
        A, B, G = 1, 20, 22

    lines = [f"{n} {K}", f"{A} {B} {G}"]
    lines.append(" ".join(str(w) for w in weights))
    lines.append(" ".join(str(p) for p in prev))
    pairs = sorted(edges.items())
    lines.append(str(len(pairs)))
    for (u, v), c in pairs:
        lines.append(f"{u} {v} {c}")
    return "\n".join(lines) + "\n"


def main():
    tid = int(sys.argv[1])
    sys.stdout.write(build(tid))


if __name__ == "__main__":
    main()
