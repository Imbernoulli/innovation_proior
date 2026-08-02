# TIER: trivial
"""Only robot 0 ever moves: one order per trip, straight there and back,
never charges. Every other robot's stop list is empty. Deliberately weak --
matches the evaluator's own reference-baseline construction."""
import sys, json
from collections import deque


def adjacency(nodes, edges):
    adj = {n["id"]: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def bfs_tree(adj, root=0):
    parent = {root: None}
    depth = {root: 0}
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                depth[v] = depth[u] + 1
                dq.append(v)
    return parent, depth


def main():
    inst = json.load(sys.stdin)
    nodes = inst["nodes"]; edges = inst["edges"]; orders = inst["orders"]
    R = inst["R"]
    adj = adjacency(nodes, edges)
    parent, depth = bfs_tree(adj, 0)
    orders_sorted = sorted(orders, key=lambda o: (o["release"], depth[o["shelf"]], o["id"]))
    routes = [[] for _ in range(R)]
    for o in orders_sorted:
        routes[0].append({"node": o["shelf"], "hold": 0})
        routes[0].append({"node": 0, "hold": 0})
    priority = list(range(R))
    print(json.dumps({"priority": priority, "routes": routes}))


if __name__ == "__main__":
    main()
