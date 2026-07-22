# TIER: strong
"""The insight: cancellation must be planned in the TRUE, time-varying
stiffness metric, not in the struts' raw ratings. Because stiffness is a
per-connected-component running count, the problem decomposes exactly along
connected components (a weld in one component never changes another
component's stiffness). For each component we run a small seeded local
search that simulates the EXACT stiffness-discounted accumulation while
jointly searching over weld order and side -- so it can discover, e.g.,
that firing a mirrored strut through a *different* matched-stiffness path
(rather than immediately after its geometric partner) is what actually
cancels a joint's displacement."""
import random
import sys


def find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def union(p, wc, a, b):
    ra, rb = find(p, a), find(p, b)
    if ra == rb:
        wc[ra] += 1
        return
    merged = wc[ra] + wc[rb] + 1
    p[rb] = ra
    wc[ra] = merged


def sim_component(k_nodes, local_edges, order, sides):
    """local_edges: list of (lu, lw, eff) indexed by local edge id.
    order: permutation of 0..len(local_edges)-1. sides aligned."""
    p = list(range(k_nodes))
    wc = [0] * k_nodes
    disp = [0.0] * k_nodes
    for ei, s in zip(order, sides):
        lu, lw, eff = local_edges[ei]
        ru, rw = find(p, lu), find(p, lw)
        su, sw = wc[ru], wc[rw]
        disp[lu] += s * eff / (1.0 + su)
        disp[lw] -= s * eff / (1.0 + sw)
        union(p, wc, lu, lw)
    return max(abs(x) for x in disp) if disp else 0.0


def local_search(k_nodes, local_edges, rng, restarts=16, iters=350):
    ne = len(local_edges)
    if ne == 0:
        return [], []

    def greedy_init():
        ledger = [0.0] * k_nodes
        order = list(range(ne))
        sides = []
        for ei in order:
            lu, lw, eff = local_edges[ei]
            cu_p, cw_p = ledger[lu] + eff, ledger[lw] - eff
            cu_m, cw_m = ledger[lu] - eff, ledger[lw] + eff
            if max(abs(cu_p), abs(cw_p)) <= max(abs(cu_m), abs(cw_m)):
                sides.append(1); ledger[lu], ledger[lw] = cu_p, cw_p
            else:
                sides.append(-1); ledger[lu], ledger[lw] = cu_m, cw_m
        return order, sides

    best_order, best_sides = greedy_init()
    best_val = sim_component(k_nodes, local_edges, best_order, best_sides)

    for r in range(restarts):
        if r == 0:
            order = list(best_order)
            sides = list(best_sides)
        else:
            order = list(range(ne))
            rng.shuffle(order)
            sides = [rng.choice((1, -1)) for _ in range(ne)]
        val = sim_component(k_nodes, local_edges, order, sides)
        for _ in range(iters):
            if ne >= 2 and rng.random() < 0.5:
                i, j = rng.sample(range(ne), 2)
                order[i], order[j] = order[j], order[i]
                nval = sim_component(k_nodes, local_edges, order, sides)
                if nval <= val:
                    val = nval
                else:
                    order[i], order[j] = order[j], order[i]
            else:
                i = rng.randrange(ne)
                sides[i] = -sides[i]
                nval = sim_component(k_nodes, local_edges, order, sides)
                if nval <= val:
                    val = nval
                else:
                    sides[i] = -sides[i]
        if val < best_val:
            best_val = val
            best_order, best_sides = list(order), list(sides)

    return best_order, best_sides


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it)); w = int(next(it)); eff = int(next(it))
        edges.append((u, w, eff))

    # partition into connected components (deterministic, input-order driven)
    p = list(range(n))

    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for u, w, _ in edges:
        ru, rw = f(u), f(w)
        if ru != rw:
            p[rw] = ru

    comp_of = [f(v) for v in range(n)]
    comp_ids = sorted(set(comp_of))
    comp_index = {c: idx for idx, c in enumerate(comp_ids)}

    comp_nodes = [[] for _ in comp_ids]
    node_local = [0] * n
    for v in range(n):
        cidx = comp_index[comp_of[v]]
        node_local[v] = len(comp_nodes[cidx])
        comp_nodes[cidx].append(v)

    comp_edges = [[] for _ in comp_ids]   # (lu, lw, eff)
    comp_edge_gidx = [[] for _ in comp_ids]  # global edge index, aligned
    for gi, (u, w, eff) in enumerate(edges):
        cidx = comp_index[comp_of[u]]
        comp_edges[cidx].append((node_local[u], node_local[w], eff))
        comp_edge_gidx[cidx].append(gi)

    rng = random.Random(20260720)

    final_order = []
    final_sides = []
    for cidx in range(len(comp_ids)):
        k_nodes = len(comp_nodes[cidx])
        local_edges = comp_edges[cidx]
        order, sides = local_search(k_nodes, local_edges, rng)
        gidx = comp_edge_gidx[cidx]
        for oi, s in zip(order, sides):
            final_order.append(gidx[oi])
            final_sides.append(s)

    out = [str(m)]
    for e, s in zip(final_order, final_sides):
        out.append(f"{e} {s}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
