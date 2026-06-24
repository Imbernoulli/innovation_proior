import sys
import heapq

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    edges = []
    # adjacency by station: list of (neighbor, line)
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        c = int(data[idx]); idx += 1
        adj[u].append((v, c))
        adj[v].append((u, c))
        edges.append((u, v, c))

    if n - 1 == 0:
        print(0)
        return

    # State = (station, last_line). last_line = None means "no line ridden yet".
    # Cost of traversing an edge on line c:
    #   if last_line is None -> 0 (boarding first line is free)
    #   elif c == last_line -> 0
    #   else -> 1 (transfer)
    # We want min total transfers to reach station n-1 (any last_line).
    # Dijkstra over (cost, station, last_line). last_line encoded: -1 for None.
    start = 0
    INF = float('inf')
    # dist dict keyed by (station, last_line)
    dist = {}
    pq = [(0, start, -1)]
    dist[(start, -1)] = 0
    best = INF
    while pq:
        d, u, last = heapq.heappop(pq)
        if d > dist.get((u, last), INF):
            continue
        if u == n - 1:
            best = min(best, d)
            # don't break; but since Dijkstra pops in order, first time we hit target is min
            # we can break safely
            print(best)
            return
        for (w, c) in adj[u]:
            if last == -1:
                nd = d  # first boarding free
            elif c == last:
                nd = d
            else:
                nd = d + 1
            key = (w, c)
            if nd < dist.get(key, INF):
                dist[key] = nd
                heapq.heappush(pq, (nd, w, c))

    print(-1)

main()
