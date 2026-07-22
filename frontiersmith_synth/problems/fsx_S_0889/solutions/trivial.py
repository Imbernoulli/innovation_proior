# TIER: trivial
"""Core-blind topological order: ascending-id Kahn's algorithm. Identical in
spirit to the checker's own internal baseline construction -- ignores both
the pipeline hazards and the cache entirely, just emits *a* valid order."""
import sys
import heapq


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)

    def nx():
        return next(it)

    N = int(nx())
    M = int(nx())
    nx()  # S, unused

    for _ in range(N):
        oid = int(nx())
        t = nx()
        if t != "A":
            nx()  # addr

    preds = [[] for _ in range(N)]
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(nx())
        v = int(nx())
        preds[v].append(u)
        adj[u].append(v)

    indeg = [len(preds[i]) for i in range(N)]
    ready = [i for i in range(N) if indeg[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        o = heapq.heappop(ready)
        order.append(o)
        for w in adj[o]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(ready, w)

    sys.stdout.write(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
