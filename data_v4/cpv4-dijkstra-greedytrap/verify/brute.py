import sys
from heapq import heappush, heappop

# Independent brute force.
# State space: (node, last_color). We do a full Bellman-Ford-style relaxation /
# label-correcting BFS over ALL states with NO Dijkstra ordering assumption:
# repeatedly relax every edge from every reachable state until no distance improves.
# This is the obviously-correct shortest path on the expanded state graph; it does
# not rely on the non-negativity argument that Dijkstra needs, it just iterates to a
# fixpoint (edge weights here are non-negative so it terminates).

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    S = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        c = int(data[idx]); idx += 1
        w = int(data[idx]); idx += 1
        adj[u].append((v, c, w))

    INF = float('inf')
    # dist[(node, last_color)] ; last_color 0 = sentinel start
    dist = {}
    dist[(1, 0)] = 0
    # label-correcting: keep a queue of states whose distance changed.
    from collections import deque
    inq = set()
    q = deque()
    q.append((1, 0))
    inq.add((1, 0))
    while q:
        node, lc = q.popleft()
        inq.discard((node, lc))
        d = dist[(node, lc)]
        for (v, c, w) in adj[node]:
            surcharge = S if (lc != 0 and c != lc) else 0
            nd = d + w + surcharge
            key = (v, c)
            if nd < dist.get(key, INF):
                dist[key] = nd
                if key not in inq:
                    inq.add(key)
                    q.append(key)

    ans = INF
    for (node, lc), d in dist.items():
        if node == n:
            ans = min(ans, d)
    print(-1 if ans == INF else ans)

main()
