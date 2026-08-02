# TIER: strong
"""The reservation-discipline insight, as a discrete-event dispatcher: wake
the robot with the earliest free virtual clock; batch up to K currently-
released orders per round trip (anchored at whichever shelf has the most
ready orders right now, topped up from nearby shelves -- sweeping outward in
one pass divides shared aisle-edge traffic by roughly K instead of one round
trip per order); charge PROACTIVELY (detour to the nearest charger and top
up BEFORE a trip that would run the battery below what it needs, rather than
after stranding); and retire a robot once its next trip would finish past the
horizon so its remaining candidates go to someone still active. This spends
individual robots' time (charging detours) to raise fleet-wide throughput --
the opposite of routing every robot on its own myopic shortest path with no
regard for timing, batching or battery."""
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


def path(parent, depth, u, v):
    pu, pv = [u], [v]
    cu, cv = u, v
    while depth[cu] > depth[cv]:
        cu = parent[cu]; pu.append(cu)
    while depth[cv] > depth[cu]:
        cv = parent[cv]; pv.append(cv)
    while cu != cv:
        cu = parent[cu]; pu.append(cu)
        cv = parent[cv]; pv.append(cv)
    return pu + pv[-2::-1]


def plen(parent, depth, u, v):
    return len(path(parent, depth, u, v)) - 1


def main():
    inst = json.load(sys.stdin)
    nodes = inst["nodes"]; edges = inst["edges"]; orders = inst["orders"]
    R = inst["R"]; K = inst["K"]; B_max = inst["B_max"]; T = inst["T"]
    adj = adjacency(nodes, edges)
    parent, depth = bfs_tree(adj, 0)
    charge_nodes = [n["id"] for n in nodes if n["kind"] == "charge"]
    charge_rate = {n["id"]: n.get("rate", 1) for n in nodes if n["kind"] == "charge"}

    def nearest_charge(u):
        if not charge_nodes:
            return None
        return min(charge_nodes, key=lambda c: plen(parent, depth, u, c))

    def trip_cost(shelves, from_pos):
        p = from_pos; c = 0
        for s in shelves:
            c += plen(parent, depth, p, s); p = s
        c += plen(parent, depth, p, 0)
        return c

    remaining = list(orders)
    routes = [[] for _ in range(R)]
    free_time = [0.0] * R
    pos = [0] * R
    battery = [B_max] * R
    INF = float("inf")

    while remaining:
        r = min(range(R), key=lambda i: free_time[i])
        if free_time[r] == INF:
            break
        clock = free_time[r]
        candidates = [o for o in remaining if o["release"] <= clock]
        if not candidates:
            nxt = min(o["release"] for o in remaining)
            nxt = max(clock, nxt)
            free_time[r] = nxt if nxt <= T else INF
            continue

        start_pos = pos[r]; start_batt = battery[r]
        # nearest-first (from wherever this robot currently is), largest-
        # affordable-prefix batching: try the K nearest currently-released
        # candidates; if unaffordable even after a full recharge, trim the
        # farthest and retry down to one shelf -- never commit to a sweep
        # this robot cannot actually complete.
        cand_sorted = sorted(
            candidates, key=lambda o: (plen(parent, depth, start_pos, o["shelf"]), o["release"], o["id"])
        )[:max(K, 1)]

        best = None
        for size in range(len(cand_sorted), 0, -1):
            chunk = cand_sorted[:size]
            shelves = sorted({o["shelf"] for o in chunk}, key=lambda s: plen(parent, depth, start_pos, s))
            cost_direct = trip_cost(shelves, start_pos)
            if start_batt >= cost_direct:
                best = (chunk, shelves, cost_direct, None)
                break
            if charge_nodes:
                c = nearest_charge(start_pos)
                d_to_c = plen(parent, depth, start_pos, c)
                if start_batt >= d_to_c:
                    cost_from_c = trip_cost(shelves, c)
                    if B_max >= cost_from_c:
                        best = (chunk, shelves, cost_from_c, c)
                        break
        if best is None:
            free_time[r] = INF
            continue

        chunk, shelves, cost, charge_c = best
        stops = []
        extra_clock = 0.0
        new_batt = start_batt
        if charge_c is not None:
            d_to_c = plen(parent, depth, start_pos, charge_c)
            rate = max(1, charge_rate.get(charge_c, 1))
            batt_at_c = start_batt - d_to_c
            deficit = max(0, B_max - batt_at_c)
            hold = -(-deficit // rate) if deficit > 0 else 0
            stops.append({"node": charge_c, "hold": hold})
            new_batt = min(B_max, batt_at_c + hold * rate)
            extra_clock = d_to_c + hold

        finish = clock + extra_clock + cost
        if finish > T:
            free_time[r] = INF
            continue

        for s in shelves:
            stops.append({"node": s, "hold": 0})
        stops.append({"node": 0, "hold": 0})
        routes[r].extend(stops)
        battery[r] = new_batt - cost
        pos[r] = 0
        free_time[r] = finish

        ids_chosen = {o["id"] for o in chunk}
        remaining = [o for o in remaining if o["id"] not in ids_chosen]

    priority = list(range(R))
    print(json.dumps({"priority": priority, "routes": routes}))


if __name__ == "__main__":
    main()
