# TIER: strong
"""The insight: total cost decomposes additively over PER-KEY contributions
once you fix everyone else's shard (imbalance is quadratic in each shard's
load, cross-shard cost is a sum over incident transaction edges, migration
cost is per key) -- so moving ONE key between shards has a cheaply
computable marginal delta, which is the classic exchange-argument move used
in graph-partitioning local search (Kernighan-Lin style). This lets the
solver directly trade "how uniform are the shard sizes" against "how many
heavy transaction edges get cut" against "how much would moving this key
away from its previous shard cost" -- something a size-uniformity-only
heuristic (bin packing / hashing) cannot express, because it never looks at
the transaction graph or the previous assignment at all.

Initialization:
  * If most keys already have a previous-epoch shard, START from that
    assignment (stability is "free" unless local search finds it worth the
    migration cost to move).
  * Otherwise (cold start) greedily CONTRACT the heaviest transaction edges
    first, growing shard-sized clusters around them -- co-locating keys that
    transact together, exactly the innovation hook -- then place any
    unconnected leftover keys onto the least-loaded shard.

Refinement: repeated single-key best-improvement moves (bounded number of
deterministic passes) using exact incremental deltas, converging to a local
optimum of the *actual* three-term cost, not merely of size balance.
"""
import sys


def read_instance():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    n = int(nxt())
    K = int(nxt())
    A = float(nxt())
    B = float(nxt())
    G = float(nxt())
    weights = [int(nxt()) for _ in range(n)]
    prev = [int(nxt()) for _ in range(n)]
    m = int(nxt())
    edges = []
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u = int(nxt())
        v = int(nxt())
        c = float(nxt())
        edges.append((u, v, c))
        adj[u].append((v, c))
        adj[v].append((u, c))
    return n, K, A, B, G, weights, prev, edges, adj


def cold_start(n, K, weights, edges):
    total = sum(weights)
    cap = (total / K) * 1.6 + 1.0
    order = sorted(range(len(edges)), key=lambda i: (-edges[i][2], edges[i][0], edges[i][1]))
    assign = [None] * n
    loads = [0.0] * K

    def best_shard_for_weight(w):
        return min(range(K), key=lambda s: (loads[s] + w, s))

    for idx in order:
        u, v, c = edges[idx]
        au, av = assign[u], assign[v]
        if au is None and av is None:
            s = min(range(K), key=lambda s: (max(0.0, loads[s] + weights[u] + weights[v] - cap), loads[s], s))
            assign[u] = s
            assign[v] = s
            loads[s] += weights[u] + weights[v]
        elif au is None:
            s = av if loads[av] + weights[u] <= cap else best_shard_for_weight(weights[u])
            assign[u] = s
            loads[s] += weights[u]
        elif av is None:
            s = au if loads[au] + weights[v] <= cap else best_shard_for_weight(weights[v])
            assign[v] = s
            loads[s] += weights[v]
        # else: both already placed, leave as-is (local search fixes it later)

    for i in range(n):
        if assign[i] is None:
            s = best_shard_for_weight(weights[i])
            assign[i] = s
            loads[s] += weights[i]
    return assign


def prev_start(n, K, weights, prev, adj):
    assign = [None] * n
    loads = [0.0] * K
    for i in range(n):
        if prev[i] != -1:
            assign[i] = prev[i]
            loads[prev[i]] += weights[i]
    for i in range(n):
        if assign[i] is not None:
            continue
        best_s, best_w = None, -1.0
        for (j, c) in adj[i]:
            if assign[j] is not None and c > best_w:
                best_s, best_w = assign[j], c
        if best_s is None:
            best_s = min(range(K), key=lambda s: (loads[s] + weights[i], s))
        assign[i] = best_s
        loads[best_s] += weights[i]
    return assign


def local_search(n, K, A, B, G, weights, prev, adj, assign, max_passes=15):
    total = sum(weights)
    avg = total / K
    loads = [0.0] * K
    for i in range(n):
        loads[assign[i]] += weights[i]

    for _ in range(max_passes):
        improved = False
        for i in range(n):
            s_old = assign[i]
            w_i = weights[i]
            best_s, best_delta = s_old, 0.0

            # cost of edges to each shard s if key i were placed in s:
            # sum over neighbors j of c * (1 if assign[j] != s else 0)
            neigh_cost = [0.0] * K
            total_edge_w = 0.0
            for (j, c) in adj[i]:
                total_edge_w += c
                neigh_cost[assign[j]] += c  # cost saved (i.e. NOT cut) if s == assign[j]

            load_old = loads[s_old]
            for s in range(K):
                if s == s_old:
                    continue
                load_new = loads[s]
                imb_old = (load_old - avg) ** 2 + (load_new - avg) ** 2
                imb_new = (load_old - w_i - avg) ** 2 + (load_new + w_i - avg) ** 2
                d_imb = imb_new - imb_old

                cross_old = total_edge_w - neigh_cost[s_old]
                cross_new = total_edge_w - neigh_cost[s]
                d_cross = cross_new - cross_old

                mig_old = w_i if (prev[i] != -1 and s_old != prev[i]) else 0.0
                mig_new = w_i if (prev[i] != -1 and s != prev[i]) else 0.0
                d_mig = mig_new - mig_old

                delta = A * d_imb + B * d_cross + G * d_mig
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_s = s

            if best_s != s_old:
                loads[s_old] -= w_i
                loads[best_s] += w_i
                assign[i] = best_s
                improved = True
        if not improved:
            break
    return assign


def main():
    n, K, A, B, G, weights, prev, edges, adj = read_instance()

    known = sum(1 for p in prev if p != -1)
    if known >= max(1, n // 3):
        assign = prev_start(n, K, weights, prev, adj)
    else:
        assign = cold_start(n, K, weights, edges)

    assign = local_search(n, K, A, B, G, weights, prev, adj, assign)

    print(" ".join(str(a) for a in assign))


if __name__ == "__main__":
    main()
