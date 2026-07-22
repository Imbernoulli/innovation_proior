# TIER: strong
"""Lagrangian relaxation of the single coupling (backbone-budget) constraint.

Price the backbone at lambda >= 0: add lambda*weight to the cost of every
trunk edge. At a FIXED lambda the commodities are completely decoupled -- each
commodity's subproblem is an independent min-cost-flow instance, solved with a
generic successive-shortest-augmenting-path solver (works for any small DAG,
not hardcoded to "trunk vs bypass"). We search lambda with a subgradient-style
update: if aggregate backbone usage exceeds the budget, raise the price; if it
undershoots, lower it; halve the step whenever the violation's sign flips
(classic scalar dual ascent). Every FEASIBLE iterate we see is a candidate
primal solution; we keep the cheapest one (primal recovery).

Because the resulting per-commodity choice is "bang-bang" at any fixed lambda
(trunk maxed out below the commodity's own threshold price, zero above it), the
best iterate found this way can still leave a little backbone budget unused
right at the margin. We close that gap with a final pass that tops up
whichever not-yet-maxed commodities have the highest dual-revealed marginal
value (cost saved per backbone unit), using only the leftover budget -- the
standard primal-repair companion to Lagrangian relaxation.
"""
import sys
from collections import deque


def mcmf(n, edges, s, t, need):
    """edges: list of (u, v, cap, cost) -- cost may be float (priced)."""
    graph = [[] for _ in range(n)]
    arc_of = [None] * len(edges)
    for i, (u, v, cap, cost) in enumerate(edges):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        arc_of[i] = (u, len(graph[u]) - 1)
    flow = 0
    INF = float("inf")
    while flow < need:
        dist = [INF] * n
        dist[s] = 0.0
        inq = [False] * n
        pv = [None] * n
        dq = deque([s])
        inq[s] = True
        while dq:
            u = dq.popleft()
            inq[u] = False
            du = dist[u]
            for idx, arc in enumerate(graph[u]):
                v, cap, cost, _rev = arc
                if cap > 0 and du + cost < dist[v] - 1e-9:
                    dist[v] = du + cost
                    pv[v] = (u, idx)
                    if not inq[v]:
                        dq.append(v)
                        inq[v] = True
        if dist[t] == INF:
            break
        push = need - flow
        v = t
        while v != s:
            u, idx = pv[v]
            push = min(push, graph[u][idx][1])
            v = u
        v = t
        while v != s:
            u, idx = pv[v]
            graph[u][idx][1] -= push
            graph[v][graph[u][idx][3]][1] += push
            v = u
        flow += push
    flows = []
    for i in range(len(edges)):
        au, aidx = arc_of[i]
        flows.append(edges[i][2] - graph[au][aidx][1])
    return flow, flows


def read_instance():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(toks[p]); p += 1; return v

    K = nxt()
    commodities = []
    for _ in range(K):
        n = nxt(); m = nxt(); s = nxt(); t = nxt(); d = nxt()
        edges = []
        for _ in range(m):
            u = nxt(); v = nxt(); cap = nxt(); cost = nxt(); shared = nxt(); weight = nxt()
            edges.append((u, v, cap, cost, shared, weight))
        commodities.append({"n": n, "s": s, "t": t, "d": d, "edges": edges})
    C = nxt()
    return commodities, C


def solve_commodity_at_price(c, lam):
    """Solve commodity c's min-cost-flow subproblem with trunk edges priced at
    lambda per backbone unit. Returns (flows, usage, true_cost)."""
    priced = []
    for (u, v, cap, cost, shared, weight) in c["edges"]:
        pc = cost + (lam * weight if shared else 0.0)
        priced.append((u, v, cap, pc))
    _achieved, flows = mcmf(c["n"], priced, c["s"], c["t"], c["d"])
    usage = 0
    true_cost = 0
    for (edge, f) in zip(c["edges"], flows):
        if edge[4]:
            usage += f * edge[5]
        true_cost += f * edge[3]
    return flows, usage, true_cost


def evaluate(commodities, lam):
    all_flows, total_usage, total_cost = [], 0, 0
    for c in commodities:
        flows, usage, cost = solve_commodity_at_price(c, lam)
        all_flows.append(flows)
        total_usage += usage
        total_cost += cost
    return all_flows, total_usage, total_cost


def marginal_value(c):
    """Cost saved per backbone unit if one flow unit rides the trunk instead
    of the cheapest DIRECT source->sink fallback route (not any non-trunk
    edge -- a completion-only edge like the trunk's own toll hop is not a
    usable route by itself)."""
    trunk = next(e for e in c["edges"] if e[4])
    direct_costs = [e[3] for e in c["edges"]
                     if not e[4] and e[0] == c["s"] and e[1] == c["t"]]
    bypass_cost = min(direct_costs) if direct_costs else trunk[3]
    return (bypass_cost - trunk[3]) / trunk[5] if trunk[5] else 0.0


def main():
    commodities, C = read_instance()

    lam = 0.0
    step = 64.0
    prev_sign = 0
    best_cost = None
    best_flows = None

    for _it in range(80):
        all_flows, usage, cost = evaluate(commodities, lam)
        if usage <= C and (best_cost is None or cost < best_cost):
            best_cost, best_flows = cost, all_flows
        violation = usage - C
        if violation == 0:
            break
        sign = 1 if violation > 0 else -1
        if prev_sign != 0 and sign != prev_sign:
            step *= 0.5
            if step < 1e-6:
                break
        lam = max(0.0, lam + step * sign)
        prev_sign = sign

    if best_flows is None:
        # Guaranteed-feasible fallback: a very high price drives every
        # commodity's trunk usage toward zero.
        best_flows, _usage, best_cost = evaluate(commodities, 1e6)

    # Primal-repair top-up: use any leftover backbone budget on the
    # highest-marginal-value commodities that are not yet at their trunk cap.
    trunk_idx = []
    for c in commodities:
        for i, e in enumerate(c["edges"]):
            if e[4]:
                trunk_idx.append(i)
                break

    def used_backbone(flows_list):
        tot = 0
        for c, flows, ti in zip(commodities, flows_list, trunk_idx):
            e = c["edges"][ti]
            tot += flows[ti] * e[5]
        return tot

    slack = C - used_backbone(best_flows)
    order = sorted(range(len(commodities)), key=lambda i: -marginal_value(commodities[i]))
    for i in order:
        if slack <= 0:
            break
        c = commodities[i]
        ti = trunk_idx[i]
        e = c["edges"][ti]
        cur_trunk = best_flows[i][ti]
        cap_trunk, weight = e[2], e[5]
        headroom = cap_trunk - cur_trunk
        if headroom <= 0 or weight <= 0:
            continue
        add = min(headroom, slack // weight)
        if add <= 0:
            continue
        new_trunk_cap = cur_trunk + add
        priced = []
        for (u, v, cap, cost, shared, weight2) in c["edges"]:
            cap_here = min(cap, new_trunk_cap) if shared else cap
            priced.append((u, v, cap_here, cost))
        _achieved, new_flows = mcmf(c["n"], priced, c["s"], c["t"], c["d"])
        new_usage = new_flows[ti] * weight
        if new_usage >= cur_trunk * weight:
            best_flows[i] = new_flows
            slack = C - used_backbone(best_flows)

    out = []
    for flows in best_flows:
        out.extend(str(f) for f in flows)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
