# TIER: greedy
"""Textbook fixed-latency list scheduling: round-robin across independent
dependency chains to keep both issue slots busy and hide hazard stalls.
This is cache-OBLIVIOUS -- it never looks at an address or at S, it just
assumes every load behaves the same (a constant latency) and interleaves
chains fairly to maximize apparent instruction-level parallelism."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)

    def nx():
        return next(it)

    N = int(nx())
    M = int(nx())
    nx()  # S, unused -- greedy is cache-oblivious

    for _ in range(N):
        oid = int(nx())
        t = nx()
        if t != "A":
            nx()  # addr, ignored

    preds = [[] for _ in range(N)]
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(nx())
        v = int(nx())
        preds[v].append(u)
        adj[u].append(v)

    # Union-find to recover the independent dependency chains ("streams").
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for u in range(N):
        for v in adj[u]:
            union(u, v)

    # One global topological order, then split by component preserving
    # relative order -- valid per-component order since edges never cross
    # components.
    indeg = [len(preds[i]) for i in range(N)]
    ready = [i for i in range(N) if indeg[i] == 0]
    ready.sort()
    global_order = []
    import heapq
    heapq.heapify(ready)
    while ready:
        o = heapq.heappop(ready)
        global_order.append(o)
        for w in adj[o]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(ready, w)

    comp_seq = {}
    comp_min_id = {}
    for node in global_order:
        r = find(node)
        comp_seq.setdefault(r, []).append(node)
        comp_min_id[r] = min(comp_min_id.get(r, node), node)

    comps = sorted(comp_seq.keys(), key=lambda r: comp_min_id[r])
    seqs = [comp_seq[r] for r in comps]

    order = []
    max_len = max((len(s) for s in seqs), default=0)
    for r in range(max_len):
        for s in seqs:
            if r < len(s):
                order.append(s[r])

    sys.stdout.write(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
