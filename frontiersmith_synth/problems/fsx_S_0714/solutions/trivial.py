# TIER: trivial
"""Cache-blind topological order: ascending-id Kahn's algorithm. Identical to
the checker's own internal baseline construction -- ignores the cart
entirely."""
import sys
import heapq


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    int(next(it))  # K, unused

    op_ids = []
    for _ in range(N):
        oid = int(next(it))
        k = int(next(it))
        for _ in range(k):
            next(it)
        next(it)  # write addr
        op_ids.append(oid)

    indeg = {o: 0 for o in op_ids}
    adj = {o: [] for o in op_ids}
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        indeg[v] += 1
        adj[u].append(v)

    ready = [o for o in op_ids if indeg[o] == 0]
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
