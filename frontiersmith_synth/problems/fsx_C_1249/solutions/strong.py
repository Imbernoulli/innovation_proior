# TIER: strong
# Insight: link BUDGET and traffic OPS are different currencies. Pay the
# cheapest possible price (a bare ring, cost N) to buy connectivity, then
# spend every remaining budget unit on the express link with the best
# "traffic saved per budget dollar" (ROI), re-measuring shortest-path
# distances after each addition (adding one link can shorten many other
# pairs' routes too, and relieves the backbone edges those flows used to
# pile onto). This directly targets the measured heavy edges instead of
# building a traffic-blind local mesh, so it fixes hop count AND congestion
# for the pairs that actually matter.
import sys
from collections import deque

def ring_cost(i, j, N):
    d = abs(i - j)
    return min(d, N - d)

def all_pairs_dist(N, adj):
    dist = [[0] * N for _ in range(N)]
    for src in range(N):
        d = [-1] * N
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] == -1:
                    d[v] = d[u] + 1
                    q.append(v)
        dist[src] = d
    return dist

def build_adj(N, edges):
    adj = [[] for _ in range(N)]
    for (u, v) in edges:
        adj[u].append(v); adj[v].append(u)
    for a in adj:
        a.sort()
    return adj

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); L_max = int(next(it))
    next(it); next(it)  # CAP, STALL_COST -- not needed for the ROI estimate
    T = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            T[i][j] = int(next(it))

    edges = [(i, (i + 1) % N) for i in range(N)]
    edges = [(min(a, b), max(a, b)) for (a, b) in edges]
    edge_set = set(edges)
    spent = N
    budget_left = L_max - N

    for _ in range(N):  # bounded re-optimization rounds
        if budget_left <= 0:
            break
        adj = build_adj(N, edges)
        dist = all_pairs_dist(N, adj)
        best = None  # (roi, cost, i, j)
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in edge_set:
                    continue
                w = T[i][j] + T[j][i]
                if w <= 0:
                    continue
                c = ring_cost(i, j, N)
                if c > budget_left:
                    continue
                d = dist[i][j]
                if d <= 1:
                    continue
                benefit = w * (d - 1)
                roi = benefit / c
                key = (roi, -c, -i, -j)
                if best is None or key > best[0]:
                    best = (key, c, i, j)
        if best is None:
            break
        _, c, i, j = best
        edges.append((i, j))
        edge_set.add((i, j))
        spent += c
        budget_left -= c

    out = [str(len(edges))]
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
