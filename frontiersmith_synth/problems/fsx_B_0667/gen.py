#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE relay-clock upgrade instance to stdout.

Instance = two densely-synced relay clusters (complete graphs, base cable
weight 1 on every existing link) joined by a *few* cross-cluster cables:
one or two "trunk" cables between designated hub relays, plus a handful of
"detour" cables between ordinary relays on each side.  Every relay has a
rationed weighted-degree cap on how much cable weight may terminate at it;
the hub relays are capped almost exactly at their current load (no spare
capacity), while the detour relays are left with generous spare capacity.
An integer cable-weight budget W may be spent (in integer units) upgrading
any existing cable's weight, subject to those per-relay caps.

Seed is testId only -> fully deterministic, reproducible generator.
"""
import random
import sys


def build(test_id: int):
    rng = random.Random(20000 + 97 * test_id)

    ks = [4, 5, 6, 6, 7, 8, 9, 10, 11, 13]
    k = ks[(test_id - 1) % len(ks)]
    n = 2 * k
    left = list(range(0, k))
    right = list(range(k, 2 * k))

    edges = []
    for cluster in (left, right):
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                edges.append((cluster[i], cluster[j]))

    hub_l, hub_r = 0, k
    edges.append((hub_l, hub_r))
    hub_nodes = {hub_l, hub_r}

    second_bridge = test_id >= 6 and k >= 6
    if second_bridge:
        hub_l2, hub_r2 = 1, k + 1
        edges.append((hub_l2, hub_r2))
        hub_nodes |= {hub_l2, hub_r2}

    used = set(edges)
    left_pool = [v for v in left if v not in hub_nodes]
    right_pool = [v for v in right if v not in hub_nodes]
    rng.shuffle(left_pool)
    rng.shuffle(right_pool)

    n_detour = 2 + (test_id % 4)
    n_detour = min(n_detour, len(left_pool), len(right_pool))
    detour_nodes = set()
    for t in range(n_detour):
        u, v = left_pool[t], right_pool[t]
        if (u, v) in used or (v, u) in used:
            continue
        edges.append((u, v))
        used.add((u, v))
        detour_nodes.add(u)
        detour_nodes.add(v)

    m = len(edges)
    deg = [0] * n
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1

    hub_slack = rng.randint(1, 3)
    w_budget = 3 * k + 6 + (4 if second_bridge else 0)
    detour_slack = w_budget + rng.randint(0, 3)

    cap = [0] * n
    for i in range(n):
        if i in hub_nodes:
            cap[i] = deg[i] + hub_slack
        elif i in detour_nodes:
            cap[i] = deg[i] + detour_slack
        else:
            cap[i] = deg[i] + rng.randint(1, 4)

    return n, m, w_budget, edges, cap


def main():
    test_id = int(sys.argv[1])
    n, m, w_budget, edges, cap = build(test_id)
    out = [f"{n} {m} {w_budget}"]
    for (u, v) in edges:
        out.append(f"{u} {v}")
    out.append(" ".join(str(c) for c in cap))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
