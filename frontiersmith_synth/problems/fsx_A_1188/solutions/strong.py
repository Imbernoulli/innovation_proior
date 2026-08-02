# TIER: strong
"""
The insight: a query only pays off once it fully SEPARATES a confounded
cluster (>= g-1 of its g nodes probed) -- a lone query into an oversized
cluster is wasted exactly like a query into an already-unique node. So:

  1. Partition nodes into clusters by their (sorted) product subset --
     purely structural, read straight off the input, no hidden data needed.
  2. Singleton clusters (already unique from baseline alone) never need a
     query -- skip them for free.
  3. For every cluster of size g >= 2, finishing it costs exactly g-1
     queries and is worth its total flux. Choosing WHICH clusters to fully
     finish under budget Q is a 0/1 knapsack over clusters (cost = g-1,
     value = cluster flux) -- solved exactly by DP since the instance is
     small. This is the actual "design which perturbations to request"
     decision: it will happily skip an oversized, budget-busting cluster
     (and any node that is already uniquely determined) in favor of several
     smaller clusters that fit the SAME budget and add up to more resolved
     flux -- exactly what a per-edge weight scan (greedy) cannot see.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    testId = int(toks[idx]); idx += 1
    N = int(toks[idx]); idx += 1
    L = int(toks[idx]); idx += 1
    Q = int(toks[idx]); idx += 1
    W = int(toks[idx]); idx += 1
    weights = [int(toks[idx + i]) for i in range(N)]
    idx += N
    adj = []
    for i in range(N):
        deg = int(toks[idx]); idx += 1
        eps = tuple(int(toks[idx + j]) for j in range(deg))
        idx += deg
        adj.append(eps)
    # trailing baseline-distribution line (L floats) is not needed by this
    # strategy: the query-allocation decision is purely structural.

    clusters = {}
    for hid in range(1, N + 1):
        clusters.setdefault(adj[hid - 1], []).append(hid)

    items = []  # (cost, value, hub_ids)
    for eps, hids in clusters.items():
        g = len(hids)
        if g < 2:
            continue
        cost = g - 1
        value = sum(weights[h - 1] for h in hids)
        items.append((cost, value, hids))

    dp = [-1] * (Q + 1)
    dp[0] = 0
    choice = [[] for _ in range(Q + 1)]
    for cost, value, hids in items:
        for b in range(Q, cost - 1, -1):
            if dp[b - cost] >= 0 and dp[b - cost] + value > dp[b]:
                dp[b] = dp[b - cost] + value
                choice[b] = choice[b - cost] + [hids]
    best_b = max(range(Q + 1), key=lambda b: dp[b])
    selected = choice[best_b]

    picked = []
    for hids in selected:
        picked.extend(sorted(hids)[:len(hids) - 1])

    print(len(picked))
    print(" ".join(str(x) for x in picked))


if __name__ == "__main__":
    main()
