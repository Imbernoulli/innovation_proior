# TIER: strong
"""Reformulate as min-cost flow on the cuckoo graph: source -> each regular (cap 1) ->
its 4 (room,slot) option-nodes (cost = the ledger's own probe price) AND the shared
annex node (cost 10*f, cap s) -> sink. Solve exact min-cost max-flow (SPFA + successive
shortest augmenting paths). This is the ONE reformulation that correctly prices every
option AND the annex on equal footing, so it will proactively route a regular into the
annex whenever that is what frees the cheapest global rearrangement -- exactly the
nonlocal, cross-regular trade a per-key or per-room greedy cannot see."""
import sys
from collections import deque


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    s = int(next(it))
    r1 = [0] * n
    r2 = [0] * n
    f = [0] * n
    for i in range(n):
        r1[i] = int(next(it))
        r2[i] = int(next(it))
        f[i] = int(next(it))

    # node ids: 0 = source; 1..n = regulars; n+1..n+2m = (room,slot) nodes
    # (room r, slot t) -> n + (r-1)*2 + t ; n+2m+1 = annex; n+2m+2 = sink
    N = n + 2 * m + 3
    SRC = 0
    ANNEX = n + 2 * m + 1
    SINK = N - 1

    graph = [[] for _ in range(N)]  # each entry: [to, cap, cost, rev_index]

    def add_edge(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    def slot_node(room, slot):
        return n + (room - 1) * 2 + slot

    for i in range(n):
        add_edge(SRC, 1 + i, 1, 0)
        add_edge(1 + i, slot_node(r1[i], 1), 1, 1 * f[i])
        add_edge(1 + i, slot_node(r1[i], 2), 1, 2 * f[i])
        add_edge(1 + i, slot_node(r2[i], 1), 1, 3 * f[i])
        add_edge(1 + i, slot_node(r2[i], 2), 1, 4 * f[i])
        add_edge(1 + i, ANNEX, 1, 10 * f[i])

    for room in range(1, m + 1):
        add_edge(slot_node(room, 1), SINK, 1, 0)
        add_edge(slot_node(room, 2), SINK, 1, 0)
    add_edge(ANNEX, SINK, s, 0)

    INF = float("inf")

    def spfa():
        dist = [INF] * N
        dist[SRC] = 0
        in_queue = [False] * N
        prev_v = [-1] * N
        prev_e = [-1] * N
        dq = deque([SRC])
        in_queue[SRC] = True
        while dq:
            u = dq.popleft()
            in_queue[u] = False
            du = dist[u]
            for idx, edge in enumerate(graph[u]):
                v, cap, cost, rev = edge
                if cap > 0 and du + cost < dist[v]:
                    dist[v] = du + cost
                    prev_v[v] = u
                    prev_e[v] = idx
                    if not in_queue[v]:
                        dq.append(v)
                        in_queue[v] = True
        return dist, prev_v, prev_e

    total_flow = 0
    while total_flow < n:
        dist, prev_v, prev_e = spfa()
        if dist[SINK] == INF:
            break
        # bottleneck along the path
        d = INF
        v = SINK
        while v != SRC:
            u = prev_v[v]
            idx = prev_e[v]
            d = min(d, graph[u][idx][1])
            v = u
        v = SINK
        while v != SRC:
            u = prev_v[v]
            idx = prev_e[v]
            graph[u][idx][1] -= d
            rev = graph[u][idx][3]
            graph[v][rev][1] += d
            v = u
        total_flow += d

    # read off each regular's chosen (room,slot) or annex from residual saturation:
    # the forward edge regular->option has cap 0 iff flow was routed on it.
    # graph[1+i] holds: [0] the reverse edge back to SRC (added first), then the 5
    # forward option edges in the order they were added: (r1,1) cost1, (r1,2) cost2,
    # (r2,1) cost3, (r2,2) cost4, annex cost10 -- i.e. indices 1..5.
    out = [None] * n
    for i in range(n):
        base = 1 + i
        chosen = None
        for edge, code in zip(graph[base][1:6], (1, 2, 3, 4, 0)):
            v, cap, cost, rev = edge
            if cap == 0:
                chosen = code
                break
        out[i] = str(chosen if chosen is not None else 0)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
