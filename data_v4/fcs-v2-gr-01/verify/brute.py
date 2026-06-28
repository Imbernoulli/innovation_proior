import sys
from collections import deque

INF = float('inf')

def solve(data):
    it = iter(data)
    n = int(next(it)); m = int(next(it)); s = int(next(it)); t = int(next(it)); F = int(next(it))

    # adjacency list of residual edges: to, cap, cost, rev_index
    to = []; cap = []; cost = []; head = [[] for _ in range(n)]
    def add_edge(u, v, c, w):
        head[u].append(len(to)); to.append(v); cap.append(c); cost.append(w)
        head[v].append(len(to)); to.append(u); cap.append(0); cost.append(-w)

    for _ in range(m):
        u = int(next(it)); v = int(next(it)); c = int(next(it)); w = int(next(it))
        add_edge(u, v, c, w)

    total_flow = 0
    total_cost = 0

    # Successive shortest paths using SPFA (Bellman-Ford queue) each round.
    # SPFA handles negative edge costs directly, so no potentials are needed.
    # Obviously correct (textbook), but slow; used only as an oracle.
    while total_flow < F:
        dist = [INF] * n
        in_queue = [False] * n
        prev_e = [-1] * n
        dist[s] = 0
        dq = deque([s]); in_queue[s] = True
        while dq:
            u = dq.popleft(); in_queue[u] = False
            du = dist[u]
            for eid in head[u]:
                if cap[eid] > 0 and du + cost[eid] < dist[to[eid]]:
                    v = to[eid]
                    dist[v] = du + cost[eid]
                    prev_e[v] = eid
                    if not in_queue[v]:
                        in_queue[v] = True
                        dq.append(v)
        if dist[t] == INF:
            break  # cannot push more flow

        # bottleneck
        push = F - total_flow
        v = t
        while v != s:
            eid = prev_e[v]
            push = min(push, cap[eid])
            v = to[eid ^ 1]
        # apply
        v = t
        while v != s:
            eid = prev_e[v]
            cap[eid] -= push
            cap[eid ^ 1] += push
            v = to[eid ^ 1]
        total_flow += push
        total_cost += push * dist[t]

    if total_flow < F:
        return "IMPOSSIBLE"
    return str(total_cost)

def main():
    data = sys.stdin.read().split()
    sys.stdout.write(solve(data) + "\n")

if __name__ == "__main__":
    main()
