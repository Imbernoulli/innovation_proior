# TIER: greedy
"""The obvious first attempt: this is a DAG (edges only ever run from a
lower-id jurisdiction to a higher-id one), so just run a shortest-path DP
that MINIMIZES the total withholding, i.e. maximizes the per-hop product of
(1 - rate). This is exactly Dijkstra/DAG-shortest-path over the treaty
network's per-hop rates.

It completely ignores that substance and timing compliance are PATH-LEVEL
predicates (aggregate benefit vs. aggregate substance, total holding period
vs. a window) -- it ranks and picks a route purely on additive per-hop
rate. On the trap instances the cheapest per-hop route runs through a chain
of shell jurisdictions that individually all look attractive but whose
route AS A WHOLE fails the anti-conduit / timing rules; this solution never
checks that, so it gets zeroed out by the checker there."""
import sys


def main():
    toks = sys.stdin.read().split()
    ptr = 0
    n = int(toks[ptr]); ptr += 1
    m = int(toks[ptr]); ptr += 1
    ptr += 5           # V0 baseline_rate_bp gamma T_min T_max (ignored)
    ptr += n            # substance scores (ignored)
    adj_in = [[] for _ in range(n)]   # adj_in[v] = list of (u, rate_bp)
    for _ in range(m):
        u = int(toks[ptr]); v = int(toks[ptr + 1]); rate_bp = int(toks[ptr + 2])
        ptr += 5
        adj_in[v].append((u, rate_bp))
    # b, backbone (ignored)

    NEG = float("-inf")
    best_log = [NEG] * n
    parent = [-1] * n
    best_log[0] = 0.0
    for v in range(1, n):
        for (u, rate_bp) in adj_in[v]:
            if best_log[u] == NEG:
                continue
            keep = 1.0 - rate_bp / 10000.0
            if keep <= 0:
                continue
            import math
            cand = best_log[u] + math.log(keep)
            if cand > best_log[v]:
                best_log[v] = cand
                parent[v] = u

    target = n - 1
    path = [target]
    cur = target
    while cur != 0:
        cur = parent[cur]
        path.append(cur)
    path.reverse()

    print(len(path))
    print(" ".join(str(x) for x in path))


if __name__ == "__main__":
    main()
