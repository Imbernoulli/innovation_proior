import random
import sys


def build_tree(rnd, n, structure):
    """Return parent[] (1-indexed, parent[1]=0) for n nodes under a given topology."""
    parent = [0] * (n + 1)
    if structure == "caterpillar":
        for i in range(2, n + 1):
            parent[i] = i - 1
    elif structure == "caterpillar_branch":
        for i in range(2, n + 1):
            if rnd.random() < 0.12:
                parent[i] = rnd.randint(1, i - 1)
            else:
                parent[i] = i - 1
    elif structure == "star_clusters":
        # a handful of "bridge" nodes hang off the root; each bridge grows its
        # own caterpillar sub-branch (a cluster). Later nodes mostly extend
        # the tail of a randomly chosen existing cluster.
        n_bridges = max(3, n // 60)
        bridges = []
        nxt = 2
        for _ in range(n_bridges):
            if nxt > n:
                break
            parent[nxt] = 1
            bridges.append(nxt)
            nxt += 1
        tails = list(bridges)
        for i in range(nxt, n + 1):
            if rnd.random() < 0.08 and len(bridges) > 0:
                parent[i] = 1  # occasionally spawn a fresh cluster root
                bridges.append(i)
                tails.append(i)
            else:
                t_idx = rnd.randrange(len(tails))
                parent[i] = tails[t_idx]
                tails[t_idx] = i
    else:  # "random"
        for i in range(2, n + 1):
            parent[i] = rnd.randint(1, i - 1)
    return parent


def depths_of(parent, n):
    depth = [0] * (n + 1)
    for i in range(2, n + 1):
        depth[i] = depth[parent[i]] + 1
    return depth


def gen_costs(rnd, parent, n, regime):
    """Return (up, down) arrays, 1-indexed, up[1]=down[1]=0.
    up[i]   = labor cost when i points to parent[i] (ascent gloss)
    down[i] = labor cost when parent[i] points to i (descent gloss)
    """
    up = [0] * (n + 1)
    down = [0] * (n + 1)
    for i in range(2, n + 1):
        if regime == "balanced":
            up[i] = rnd.randint(1, 12)
            down[i] = rnd.randint(1, 12)
        elif regime == "down_cheap":
            down[i] = rnd.randint(5, 15)
            up[i] = rnd.randint(30, 70)
        elif regime == "up_cheap":
            up[i] = rnd.randint(5, 15)
            down[i] = rnd.randint(30, 70)
        elif regime == "mixed_random":
            if rnd.random() < 0.5:
                down[i] = rnd.randint(1, 3)
                up[i] = rnd.randint(80, 350)
            else:
                up[i] = rnd.randint(1, 3)
                down[i] = rnd.randint(80, 350)
        elif regime == "mixed_zones":
            zone = (i // 17) % 2
            if zone == 0:
                down[i] = rnd.randint(1, 3)
                up[i] = rnd.randint(70, 260)
            else:
                up[i] = rnd.randint(1, 3)
                down[i] = rnd.randint(70, 260)
        else:
            up[i] = rnd.randint(1, 12)
            down[i] = rnd.randint(1, 12)
    return up, down


def gen_weights(rnd, parent, n, regime, depth):
    w = [0] * (n + 1)
    if regime == "uniform":
        for i in range(1, n + 1):
            w[i] = rnd.randint(1, 5)
    elif regime == "zipf_general":
        order = list(range(1, n + 1))
        rnd.shuffle(order)
        for rank, node in enumerate(order, start=1):
            w[node] = max(1, int(2000 / (rank ** 1.1)))
    elif regime == "hot_deep":
        for i in range(1, n + 1):
            w[i] = rnd.randint(1, 4)
        max_depth = max(depth[1:n + 1])
        deep_nodes = [i for i in range(1, n + 1) if depth[i] >= int(max_depth * 0.4)]
        rnd.shuffle(deep_nodes)
        n_hot = max(2, len(deep_nodes) // 3)
        for i in deep_nodes[:n_hot]:
            w[i] = rnd.randint(60, 180)
    elif regime == "hot_clusters":
        for i in range(1, n + 1):
            w[i] = rnd.randint(1, 3)
        # find bridge-ish nodes: children directly of root
        bridgey = [i for i in range(2, n + 1) if parent[i] == 1]
        for b in bridgey:
            if rnd.random() < 0.7:
                w[b] = rnd.randint(200, 900)
        # also weight a random deep node within each cluster's descendants
        children = [[] for _ in range(n + 1)]
        for i in range(2, n + 1):
            children[parent[i]].append(i)
        for b in bridgey:
            stack = list(children[b])
            deep_pick = []
            while stack:
                u = stack.pop()
                deep_pick.append(u)
                stack.extend(children[u])
            if deep_pick and rnd.random() < 0.8:
                pick = rnd.choice(deep_pick)
                w[pick] = rnd.randint(300, 1000)
    else:
        for i in range(1, n + 1):
            w[i] = rnd.randint(1, 5)
    return w


def gen_sizes(rnd, n):
    size = [0] * (n + 1)
    size[1] = 1  # the Urtext exemplar is always cheap to keep
    for i in range(2, n + 1):
        size[i] = rnd.randint(4, 40)
    return size


def build_cluster_trap(rnd, n, n_decoys=1):
    """A dedicated 'decoy vs. many cheap wins' construction: several long
    caterpillar clusters hang off the root. Each cluster is entirely
    down_cheap (descending is cheap, ascending is expensive), so a SINGLE
    exemplar at a cluster's tail serves the whole cluster cheaply via
    descent glosses -- but only if something is willing to point that way.
    A couple of "decoy" nodes near the root carry a huge raw read-weight
    AND a huge vellum cost: a popularity-only knapsack greedily burns the
    whole budget on them, starving the many cheap, high-density cluster
    wins that a value-density search would prefer instead.
    Returns (parent, up, down, size, w, s_budget).
    """
    n_clusters = max(6, n // 45)
    cluster_len = max(4, (n - 1) // n_clusters)

    parent = [0] * (n + 1)
    up = [0] * (n + 1)
    down = [0] * (n + 1)
    size = [0] * (n + 1)
    w = [0.0] * (n + 1)
    size[1] = 1
    w[1] = 1

    clusters = []
    idx = 2
    while idx <= n:
        length = max(2, cluster_len + rnd.randint(-2, 2))
        length = min(length, n - idx + 1)
        nodes = []
        prev = 1
        for _ in range(length):
            i = idx
            parent[i] = prev
            down[i] = rnd.randint(16, 28)
            up[i] = rnd.randint(50, 150)
            size[i] = rnd.randint(22, 38)
            w[i] = rnd.randint(5, 20)
            nodes.append(i)
            prev = i
            idx += 1
            if idx > n:
                break
        clusters.append(nodes)

    decoy_size_total = 0
    for c_i in range(min(n_decoys, len(clusters))):
        pick = clusters[c_i][0]
        w[pick] = rnd.randint(3000, 6000)
        size[pick] = rnd.randint(150, 250)
        decoy_size_total += size[pick]

    efficient_tail_sizes = []
    for c_i in range(n_decoys, len(clusters)):
        nodes = clusters[c_i]
        deep = nodes[-1]
        w[deep] = rnd.randint(150, 400)
        efficient_tail_sizes.append(size[deep])

    # budget: exactly enough to tempt a popularity-only greedy into burning
    # (almost) everything on the decoy(s), while a density-aware search that
    # skips the decoy(s) can afford roughly HALF of the cheap cluster-tail
    # wins -- large enough to show a real gap, small enough to leave the
    # other half of the clusters genuinely uncovered (no saturation).
    efficient_tail_sizes.sort()
    n_afford = max(1, len(efficient_tail_sizes) // 3)
    half_efficient_budget = sum(efficient_tail_sizes[:n_afford])
    s_budget = size[1] + decoy_size_total + half_efficient_budget + rnd.randint(2, 6)
    return parent, up, down, size, w, s_budget


TESTS = [
    # (n, structure, cost_regime, weight_regime, target_k)
    dict(n=6, structure="random", cost="balanced", weight="uniform", k=1),
    dict(n=12, structure="caterpillar_branch", cost="balanced", weight="uniform", k=3),
    dict(n=60, structure="random", cost="mixed_random", weight="zipf_general", k=6),
    dict(n=150, structure="caterpillar", cost="down_cheap", weight="hot_deep", k=2),
    dict(n=250, structure="cluster_trap", cost=None, weight=None, k=None),
    dict(n=400, structure="random", cost="mixed_random", weight="zipf_general", k=8),
    dict(n=600, structure="caterpillar_branch", cost="mixed_zones", weight="hot_deep", k=5),
    dict(n=900, structure="random", cost="mixed_random", weight="zipf_general", k=12),
    dict(n=1300, structure="cluster_trap", cost=None, weight=None, k=None),
    dict(n=1800, structure="random", cost="mixed_random", weight="zipf_general", k=16),
]


def main():
    test_id = int(sys.argv[1])
    spec = TESTS[(test_id - 1) % len(TESTS)]
    rnd = random.Random(1_000_003 * test_id + 7)

    n = spec["n"]
    if spec["structure"] == "cluster_trap":
        parent, up, down, size, w, s_budget = build_cluster_trap(rnd, n)
    else:
        parent = build_tree(rnd, n, spec["structure"])
        depth = depths_of(parent, n)
        up, down = gen_costs(rnd, parent, n, spec["cost"])
        size = gen_sizes(rnd, n)
        w = gen_weights(rnd, parent, n, spec["weight"], depth)

        # budget: afford the (cheap) root plus roughly target_k-1 more
        # exemplars, picked as a mix of small/medium sizes so it is a real
        # knapsack, not a free-for-all.
        target_k = spec["k"]
        sizes_sorted = sorted(size[2:n + 1])
        extra = sizes_sorted[:max(0, target_k - 1)]
        s_budget = size[1] + sum(extra) + (target_k - 1) * 2  # small slack

    out = []
    out.append(f"{n} {s_budget}")
    out.append(f"{size[1]} {w[1]}")
    for i in range(2, n + 1):
        out.append(f"{parent[i]} {up[i]} {down[i]} {size[i]} {w[i]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
