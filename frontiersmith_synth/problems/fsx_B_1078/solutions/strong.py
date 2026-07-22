# TIER: strong
"""Exploits the Dilworth / min-path-cover duality instead of profit-first packing.

1. Build the DAG's transitive reachability (who-can-precede-whom).
2. Compute a MINIMUM CHAIN (path) COVER of the *entire* task poset via maximum
   bipartite matching on the comparability graph (Fulkerson's construction): an
   edge u-v (v reachable from u) that survives in a maximum matching means "v
   immediately follows u in some optimal chain decomposition". This is exactly
   Dilworth's theorem computed constructively -- it reveals the true, minimum
   number of parallel tracks the poset naturally decomposes into, and which
   tasks are cheap to co-locate on one track versus which permanently consume
   a whole track (antichain members).
3. Within each recovered chain, any relative-order-preserving subsequence is
   still a valid chain, so trim each chain to its top-H-profit tasks (H = the
   line horizon) while preserving order.
4. Only k tracks exist, so rank the (now trimmed) chains by realized value and
   keep the best min(k, #chains) -- correctly preferring a handful of deep,
   dense chains over many individually-tempting but mutually-incomparable
   high-profit singletons that would each waste an entire track's horizon.
"""
import sys
from collections import deque

sys.setrecursionlimit(10000)


def main():
    data = sys.stdin.buffer.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(data[p]); p += 1
        return v

    N = nxt(); M = nxt(); k = nxt(); H = nxt()
    profit = [0] * (N + 1)
    for i in range(1, N + 1):
        profit[i] = nxt()
    adj = [[] for _ in range(N + 1)]
    indeg = [0] * (N + 1)
    for _ in range(M):
        u = nxt(); v = nxt()
        adj[u].append(v)
        indeg[v] += 1

    dq = deque([i for i in range(1, N + 1) if indeg[i] == 0])
    topo = []
    indeg2 = indeg[:]
    while dq:
        u = dq.popleft()
        topo.append(u)
        for v in adj[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                dq.append(v)
    topo_pos = [0] * (N + 1)
    for idx, u in enumerate(topo):
        topo_pos[u] = idx

    reach = [0] * (N + 1)
    for u in reversed(topo):
        r = 0
        for w in adj[u]:
            r |= (1 << w) | reach[w]
        reach[u] = r

    # Bipartite comparability adjacency: u -> [v : v reachable from u].
    bi_adj = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        r = reach[u]
        v = 0
        rr = r
        while rr:
            low = rr & (-rr)
            v = low.bit_length() - 1
            bi_adj[u].append(v)
            rr ^= low

    matchR = [0] * (N + 1)

    def try_kuhn(u, visited):
        for v in bi_adj[u]:
            if not visited[v]:
                visited[v] = True
                if matchR[v] == 0 or try_kuhn(matchR[v], visited):
                    matchR[v] = u
                    return True
        return False

    # Process left endpoints in topo order (stable, deterministic, and tends to
    # produce long forward chains first).
    for u in topo:
        if bi_adj[u]:
            visited = [False] * (N + 1)
            try_kuhn(u, visited)

    next_ptr = [0] * (N + 1)
    for v in range(1, N + 1):
        if matchR[v] != 0:
            next_ptr[matchR[v]] = v

    heads = [v for v in range(1, N + 1) if matchR[v] == 0]

    chains = []
    for h in heads:
        chain = [h]
        cur = h
        while next_ptr[cur] != 0:
            cur = next_ptr[cur]
            chain.append(cur)
        chain.sort(key=lambda t: topo_pos[t])  # defensive re-order, valid since all comparable
        chains.append(chain)

    trimmed = []
    for chain in chains:
        best = sorted(chain, key=lambda t: (-profit[t], t))[:H]
        best.sort(key=lambda t: topo_pos[t])
        val = sum(profit[t] for t in best)
        trimmed.append((val, best))

    trimmed.sort(key=lambda x: (-x[0], -len(x[1])))
    picked = trimmed[:min(k, len(trimmed))]
    picked = [c for c in picked if c[1]]

    out = [str(len(picked))]
    for _, seq in picked:
        out.append(str(len(seq)) + " " + " ".join(map(str, seq)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
