# TIER: greedy
"""The obvious 'independent shortest path per robot' recipe: every robot
works, round-robins the order queue one order at a time (fetch, come
straight back, no batching), never plans a charging detour, and priority is
just robot index. No fleet-level coordination at all -- fine when the
corridor is short and lightly loaded, but floods the shared single-lane
aisle and runs batteries dry with zero recharging on the dense instances."""
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
    ri = 0
    for o in orders_sorted:
        r = ri % R
        ri += 1
        routes[r].append({"node": o["shelf"], "hold": 0})
        routes[r].append({"node": 0, "hold": 0})
    priority = list(range(R))  # naive fixed priority, no coordination
    print(json.dumps({"priority": priority, "routes": routes}))


if __name__ == "__main__":
    main()
