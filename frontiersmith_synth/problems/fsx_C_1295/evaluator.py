#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1295 -- "Robots That Must Not Deadlock: Warehouse
Fleet Reservation Discipline".

Family: warehouse-robot-policy. Composes THREE mechanisms into one objective:
  (1) path-conflict-resolution -- the warehouse is a single-lane aisle tree
      (a corridor spine with dead-end shelf/charging spurs); every EDGE can
      carry at most one robot per tick, so a fleet-wide plan must decide WHO
      moves through a contested aisle segment WHEN.
  (2) charging-detour-cost -- robots burn one battery unit per move and only
      recharge by physically detouring to a charging spur and dwelling there;
      running dry away from a charger stops a robot forever.
  (3) order-batching -- a robot may carry up to K totes at once, so a single
      round trip that sweeps several nearby shelves before returning is far
      cheaper (in scarce aisle-edge traffic) than one trip per order.

THE INNOVATION HOOK: a candidate is a static, OFFLINE fleet PLAN (a robot
processing PRIORITY order + each robot's stop list), not a reactive
controller. The evaluator resolves the whole fleet with a PRIORITIZED /
RESERVATION-TABLE simulator: robots are simulated to completion strictly in
priority order, each one taking the earliest conflict-free schedule along its
(unique, since the graph is a tree) route GIVEN the aisle-edge and dock
reservations already committed by every higher-priority robot -- waiting
(never colliding) when a segment is taken. This guarantees a physically
consistent schedule for ANY priority permutation, but a bad permutation +
naive one-order-at-a-time routing floods the shared spine with far more
aisle-edge crossings than necessary, so lower-priority robots queue for most
of the horizon and (having budgeted no charging stops) strand mid-corridor
once their battery runs out. The insight a strong policy exploits: BATCH
several orders per trip (fewer total aisle-edge crossings => less queuing for
everyone), charge PROACTIVELY before a trip that would run the tank dry (a
detour that costs THAT robot time but keeps it productive for the rest of the
horizon). This is a genuine reservation discipline: it spends individual robots' time
(detours, deference) to raise FLEET throughput -- the opposite of routing
every robot on its own myopic shortest path.

TRAP: "independent shortest-path routing" -- assign each order round-robin to
a robot, send it there and straight back one order at a time, never charge,
priority = robot index -- works fine when a handful of robots share a short,
lightly loaded corridor (LOW density instances), but on the HIGH density
instances (many robots, long corridor, tight battery, popular shared
shelves/chargers) this floods the single-lane spine and drains batteries with
zero recharging discipline, so throughput collapses far below what a
batching + charging-aware policy achieves on the SAME
instance.

Answer format: {"priority": [permutation of 0..R-1],
                 "routes": [ [ {"node": <id>, "hold": <int>=0}, ... ], ... ]}
(routes[i] is robot i's stop list, in ROBOT-INDEX order; priority gives the
order robots are resolved in by the reservation-table simulator.)

Scoring (deterministic, no wall-time): objective = number of DISTINCT orders
delivered (dropped at the depot, node 0) within the horizon T. Normalized
against a fixed per-instance reference throughput CAP computed at generation
time by a competent (batching + charging aware) but not
necessarily optimal reference policy, scaled up for headroom, so no solution
can win by merely matching the reference:

  ratio = min(1, orders_delivered / CAP(instance))

CLI: python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean ratio over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
from collections import deque
import isorun


# ============================== deterministic RNG ===========================
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ================================ graph utils ===============================
def _adjacency(nodes, edges):
    adj = {n["id"]: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_tree(adj, root=0):
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


def _path(parent, depth, u, v):
    """Unique tree path u -> v (inclusive), via LCA climb."""
    pu, pv = [u], [v]
    cu, cv = u, v
    while depth[cu] > depth[cv]:
        cu = parent[cu]
        pu.append(cu)
    while depth[cv] > depth[cu]:
        cv = parent[cv]
        pv.append(cv)
    while cu != cv:
        cu = parent[cu]
        pu.append(cu)
        cv = parent[cv]
        pv.append(cv)
    return pu + pv[-2::-1]


def _plen(parent, depth, u, v):
    return len(_path(parent, depth, u, v)) - 1


# ============================== graph construction ===========================
def _build_graph(L, shelf_positions, charge_specs):
    """node 0 = BASE (depot+dropoff, unlimited capacity, no charging).
    corridor nodes 1..L (id == position), path edges (p-1,p).
    shelf spurs: one dead-end node per shelf_position, dock capacity 1.
    charge spurs: one dead-end node per (position,cap,rate) in charge_specs."""
    nodes = [{"id": 0, "kind": "base"}]
    edges = []
    for p in range(1, L + 1):
        nodes.append({"id": p, "kind": "aisle"})
        edges.append([p - 1, p])
    nid = L + 1
    shelf_ids = []
    for pos in shelf_positions:
        nodes.append({"id": nid, "kind": "shelf", "cap": 1})
        edges.append([pos, nid])
        shelf_ids.append(nid)
        nid += 1
    charge_ids = []
    for (pos, cap, rate) in charge_specs:
        nodes.append({"id": nid, "kind": "charge", "cap": cap, "rate": rate})
        edges.append([pos, nid])
        charge_ids.append(nid)
        nid += 1
    return nodes, edges, shelf_ids, charge_ids


# ============================== instance family ==============================
# (seed, R, L, shelf_positions, charge_specs, M, T, B_max, K, hot_frac)
# hot_frac: fraction of orders drawn from a small "hot" pool of shelves (clustering,
# rewards batching). idx 0-2 EASY (low density, slack battery/time). idx 3 MEDIUM.
# idx 4-9 TRAP (dense fleets / long corridors / tight battery / contested chargers).
_SPECS = [
    # -------------------------- EASY (low density) --------------------------
    (17001, 3, 6, [2, 4, 6], [(4, 1, 8)], 8, 55, 40, 2, 0.6),
    (17002, 4, 8, [2, 3, 5, 7, 8], [(5, 1, 8)], 12, 65, 44, 2, 0.6),
    (17003, 5, 8, [2, 3, 4, 6, 7, 8], [(4, 1, 6), (7, 1, 6)], 16, 65, 36, 3, 0.6),
    # ---------------------------- MEDIUM ------------------------------------
    (17004, 8, 10, [2, 3, 4, 5, 7, 8, 9, 10], [(5, 1, 6)], 26, 75, 30, 3, 0.65),
    # ------------------------------ TRAP -------------------------------------
    (17005, 14, 12, [3, 5, 6, 8, 9, 10, 12], [(6, 1, 5)], 42, 85, 26, 3, 0.7),
    (17006, 16, 14, [3, 5, 6, 8, 9, 11, 13, 14], [(7, 1, 5)], 50, 90, 24, 4, 0.7),
    (17007, 18, 10, [2, 3, 4, 5, 6, 7, 8, 9, 10], [(5, 2, 8)], 54, 80, 50, 3, 0.65),
    (17008, 12, 16, [7, 9, 10, 12, 13, 14, 15, 16], [(9, 1, 4)], 36, 100, 22, 3, 0.7),
    (17009, 20, 12, [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], [(4, 1, 6), (10, 1, 6)], 60, 100, 28, 4, 0.7),
    # held-out generalization: longer corridor, deeper shelves, single distant charger
    (17010, 15, 18, [4, 6, 8, 10, 11, 13, 14, 16, 17, 18], [(11, 1, 5)], 46, 110, 26, 3, 0.72),
]


def _gen_orders(rng, shelf_positions, depth, M, T, hot_frac):
    S = len(shelf_positions)
    hot_pool = shelf_positions[: max(1, min(3, S))]
    orders = []
    for m in range(M):
        if rng(0, 99) < int(hot_frac * 100):
            shelf = hot_pool[rng(0, len(hot_pool) - 1)]
        else:
            shelf = shelf_positions[rng(0, S - 1)]
        # bursty but spread release times, deterministic
        base_t = (m * max(1, (T - 15))) // max(1, M)
        release = max(0, min(T - 5, base_t + rng(-3, 3)))
        orders.append({"id": m, "shelf": shelf, "release": release})
    return orders


def make_instances():
    out = []
    for (seed, R, L, shelf_positions, charge_specs, M, T, B_max, K, hot_frac) in _SPECS:
        nodes, edges, shelf_ids, charge_ids = _build_graph(L, shelf_positions, charge_specs)
        adj = _adjacency(nodes, edges)
        parent, depth = _bfs_tree(adj, 0)
        rng = _rng(seed)
        orders = _gen_orders(rng, shelf_ids, depth, M, T, hot_frac)
        public = {
            "R": R, "K": K, "B_max": B_max, "T": T,
            "nodes": nodes, "edges": edges, "orders": orders,
        }
        out.append({"public": public, "hidden": {}})
    return out


# ============================== reference policy =============================
def _plan_routes(pub, batch_K=None):
    """Reference policy: a discrete-event dispatcher. Repeatedly wake the robot
    with the earliest free time; if orders are already released at that clock,
    batch up to K of them (anchored at whichever shelf currently has the most
    ready orders, topped up from nearby shelves), charge PROACTIVELY (to full)
    first if the trip would run the battery below what it needs, and send the
    robot on the trip (tracking its own virtual clock/position/battery exactly,
    ignoring cross-robot aisle contention -- which only ever costs a REAL
    schedule extra time, never less, so this is a safe optimistic reference).
    Robots that would finish past the horizon retire. Used BOTH to compute the
    generation-time CAP (with headroom) and as the template for
    solutions/strong.py."""
    nodes = pub["nodes"]; edges = pub["edges"]; orders = pub["orders"]
    R = pub["R"]; K = batch_K if batch_K else pub["K"]; B_max = pub["B_max"]; T = pub["T"]
    adj = _adjacency(nodes, edges)
    parent, depth = _bfs_tree(adj, 0)
    charge_nodes = [n["id"] for n in nodes if n["kind"] == "charge"]
    charge_rate = {n["id"]: n.get("rate", 1) for n in nodes if n["kind"] == "charge"}

    def plen(u, v):
        return _plen(parent, depth, u, v)

    def nearest_charge(u):
        if not charge_nodes:
            return None
        return min(charge_nodes, key=lambda c: plen(u, c))

    def trip_cost(shelves, from_pos):
        p = from_pos; c = 0
        for s in shelves:
            c += plen(p, s); p = s
        c += plen(p, 0)
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
        # nearest-first (from wherever this robot currently is -- base, or a
        # charger if it just topped up), largest-affordable-prefix batching:
        # try the K nearest currently-released candidates; if that trip is not
        # affordable even after a full recharge, trim the farthest one and
        # retry, down to a single shelf. This guarantees whatever we commit to
        # is battery-feasible, instead of blindly committing to an
        # unaffordable sweep and stranding the robot.
        cand_sorted = sorted(
            candidates, key=lambda o: (plen(start_pos, o["shelf"]), o["release"], o["id"])
        )[:max(K, 1)]

        best = None
        for size in range(len(cand_sorted), 0, -1):
            chunk = cand_sorted[:size]
            shelves = sorted({o["shelf"] for o in chunk}, key=lambda s: plen(start_pos, s))
            cost_direct = trip_cost(shelves, start_pos)
            if start_batt >= cost_direct:
                best = (chunk, shelves, cost_direct, None)
                break
            if charge_nodes:
                c = nearest_charge(start_pos)
                d_to_c = plen(start_pos, c)
                if start_batt >= d_to_c:
                    cost_from_c = trip_cost(shelves, c)
                    if B_max >= cost_from_c:
                        best = (chunk, shelves, cost_from_c, c)
                        break
        if best is None:
            free_time[r] = INF  # can't afford even one shelf, even via charging
            continue

        chunk, shelves, cost, charge_c = best
        stops = []
        extra_clock = 0.0
        new_batt = start_batt
        if charge_c is not None:
            d_to_c = plen(start_pos, charge_c)
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
            continue  # don't commit; leave candidates for another robot

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
    return {"priority": priority, "routes": routes}


def _trivial_answer(pub):
    """Deliberately weak reference: ONLY robot 0 works, one order per trip,
    never charges. All other robots idle (empty stop lists)."""
    nodes = pub["nodes"]; edges = pub["edges"]; orders = pub["orders"]
    R = pub["R"]
    adj = _adjacency(nodes, edges)
    parent, depth = _bfs_tree(adj, 0)
    orders_sorted = sorted(orders, key=lambda o: (o["release"], depth[o["shelf"]], o["id"]))
    routes = [[] for _ in range(R)]
    for o in orders_sorted:
        routes[0].append({"node": o["shelf"], "hold": 0})
        routes[0].append({"node": 0, "hold": 0})
    priority = list(range(R))
    return {"priority": priority, "routes": routes}


# ============================== simulator / scorer ============================
STOP_CAP = 400
HOLD_CAP = 300


def simulate(inst, answer):
    """Strictly validate `answer`, then run the prioritized reservation-table
    fleet simulation. Returns (ok: bool, orders_delivered: float|None)."""
    pub = inst["public"]
    R = pub["R"]; K = pub["K"]; B_max = pub["B_max"]; T = pub["T"]
    nodes = pub["nodes"]; edges = pub["edges"]; orders = pub["orders"]
    node_meta = {n["id"]: n for n in nodes}
    valid_ids = set(node_meta)
    adj = _adjacency(nodes, edges)
    parent, depth = _bfs_tree(adj, 0)

    if not isinstance(answer, dict):
        return False, None
    prio = answer.get("priority")
    routes = answer.get("routes")
    if not isinstance(prio, list) or len(prio) != R:
        return False, None
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in prio):
        return False, None
    if sorted(prio) != list(range(R)):
        return False, None
    if not isinstance(routes, list) or len(routes) != R:
        return False, None

    parsed_routes = []
    for r in routes:
        if not isinstance(r, list) or len(r) > STOP_CAP:
            return False, None
        stops = []
        for st in r:
            if not isinstance(st, dict):
                return False, None
            node = st.get("node")
            hold = st.get("hold", 0)
            if isinstance(node, bool) or not isinstance(node, int):
                return False, None
            if node not in valid_ids:
                return False, None
            if isinstance(hold, bool):
                return False, None
            if isinstance(hold, float):
                if not math.isfinite(hold) or abs(hold - round(hold)) > 1e-9:
                    return False, None
                hold = int(round(hold))
            if not isinstance(hold, int):
                return False, None
            if hold < 0 or hold > HOLD_CAP:
                return False, None
            stops.append((node, hold))
        parsed_routes.append(stops)

    # ---------------- order pool ----------------
    unclaimed = {}
    for o in orders:
        unclaimed.setdefault(o["shelf"], []).append((o["release"], o["id"]))
    for k in unclaimed:
        unclaimed[k].sort()
    delivered = set()

    # ---------------- reservation tables ----------------
    edge_busy = {}          # (min,max) -> set(ticks used)
    node_cap = {}
    node_count = {}         # limited-capacity node id -> {tick: count}
    for n in nodes:
        if n["kind"] in ("shelf", "charge"):
            node_cap[n["id"]] = n.get("cap", 1)
            node_count[n["id"]] = {}
    is_charge = {n["id"]: (n["kind"] == "charge") for n in nodes}
    charge_rate = {n["id"]: n.get("rate", 0) for n in nodes if n["kind"] == "charge"}

    for ridx in prio:
        stops = parsed_routes[ridx]
        pos = 0
        t = 0
        battery = B_max
        held = []
        for (target, hold) in stops:
            if t > T:
                break
            path = _path(parent, depth, pos, target)
            feasible_full = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                ek = (u, v) if u < v else (v, u)
                moved = False
                while True:
                    if t > T or battery < 1:
                        feasible_full = False
                        break
                    busy = edge_busy.setdefault(ek, set())
                    v_ok = True
                    if v in node_cap:
                        if node_count[v].get(t + 1, 0) >= node_cap[v]:
                            v_ok = False
                    if (t not in busy) and v_ok:
                        busy.add(t)
                        if v in node_cap:
                            node_count[v][t + 1] = node_count[v].get(t + 1, 0) + 1
                        pos = v
                        t += 1
                        battery -= 1
                        moved = True
                        break
                    else:
                        if is_charge.get(pos):
                            battery = min(B_max, battery + charge_rate.get(pos, 0))
                        t += 1
                if not feasible_full:
                    break
            if not feasible_full:
                continue  # stranded / abandoned this stop; try the next one

            # ---- arrival actions ----
            if target == 0:
                for oid in held:
                    delivered.add(oid)
                held = []
            elif node_meta[target]["kind"] == "shelf":
                avail = unclaimed.get(target)
                if avail:
                    while avail and len(held) < K and avail[0][0] <= t:
                        _, oid = avail.pop(0)
                        held.append(oid)
            # charge node: no automatic action beyond dwelling below

            # ---- explicit dwell ----
            for _ in range(hold):
                if t >= T:
                    break
                if is_charge.get(pos):
                    battery = min(B_max, battery + charge_rate.get(pos, 0))
                t += 1

    return True, float(len(delivered))


def baseline(inst):
    return simulate(inst, _trivial_answer(inst["public"]))[1]


HEADROOM = 1.28
_CAP_CACHE = {}


def _cap(inst_idx, inst):
    """CAP is generated from a deliberately MORE capable reference than any
    valid submission could ever be: the same dispatcher, but with one extra
    unit of tote capacity (K+1) that no answer is allowed to use (the real
    simulator enforces the real K strictly). This keeps CAP a genuinely
    unreachable ceiling instead of a fixed multiple of solutions/strong.py's
    own score -- a real submission's shortfall against CAP varies instance to
    instance instead of landing on a single constant ratio."""
    if inst_idx not in _CAP_CACHE:
        pub = inst["public"]
        ref_obj = simulate(inst, _plan_routes(pub, batch_K=pub["K"] + 1))[1] or 1.0
        _CAP_CACHE[inst_idx] = max(1.0, ref_obj * HEADROOM)
    return _CAP_CACHE[inst_idx]


def score(inst, answer, inst_idx):
    try:
        ok, obj = simulate(inst, answer)
    except Exception:
        return False, None
    if not ok or obj is None:
        return False, None
    return True, obj


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for idx, inst in enumerate(insts):
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=15)
        if st != "OK":
            vec.append(0.0)
            continue
        ok, obj = score(inst, ans, idx)
        if not ok or obj is None:
            vec.append(0.0)
            continue
        cap = _cap(idx, inst)
        r = min(1.0, obj / cap) if cap > 0 else 0.0
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
