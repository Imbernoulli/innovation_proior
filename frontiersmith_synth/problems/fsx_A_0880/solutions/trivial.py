# TIER: trivial
"""Overcautious dispatcher: take the graph-shortest path, but flatly REFUSE any
move whose shortest path touches a single-lane shortcut at all (skip it rather
than risk a conflict). Never waits, never reroutes for moves it does attempt."""
import sys, json, heapq


def dijkstra_path(n, adj, src, dst):
    dist = [None] * n; prev = [None] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if dist[u] is not None and d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if dist[v] is None or nd < dist[v]:
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[dst] is None:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def main():
    inst = json.load(sys.stdin)
    n = inst["n_nodes"]
    adj = [[] for _ in range(n)]
    emap = {}
    for e in inst["edges"]:
        adj[e["u"]].append((e["v"], e["length"]))
        adj[e["v"]].append((e["u"], e["length"]))
        emap[frozenset((e["u"], e["v"]))] = e

    out_moves = []
    for mv in inst["moves"]:
        path = dijkstra_path(n, adj, mv["src"], mv["dst"])
        if path is None:
            continue
        uses_shared = any(emap[frozenset((path[i], path[i + 1]))]["shared"] for i in range(len(path) - 1))
        if uses_shared:
            continue
        t = mv["release"]; times = [t]
        for i in range(len(path) - 1):
            e = emap[frozenset((path[i], path[i + 1]))]
            t += e["length"]; times.append(t)
        out_moves.append({"id": mv["id"], "path": path, "times": times})

    print(json.dumps({"moves": out_moves}))


main()
