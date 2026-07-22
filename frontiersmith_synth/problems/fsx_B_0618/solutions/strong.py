# TIER: strong
"""Shape-aware tree surgery.

Insight: a fusion edge whose two local tabs are OPPOSITE (N-S or E-W) always
glues its two facets flat/coplanar -- keeping every such edge is free and
never changes any facet's position relative to the rest of its sheet, no
matter which spanning tree you eventually pick among them. Only fusion edges
whose tabs are NOT opposite are genuine 90-degree folds, and folds are where
overlap and bounding-box waste actually come from.

So: (1) union every "flat" edge first -- this deterministically partitions
the facets into rigid flat panels with a fixed internal shape, independent of
any later choice. (2) The remaining "fold" edges connect panels; group them
by which pair of panels they join (this recovers the redundant hinge
candidates the greedy pass could not tell apart). (3) Walk the panel-adjacency
tree outward from the panel containing facet 0; for every panel about to be
attached, SIMULATE every candidate fold edge in its group, reject any that
would overlap an already-placed facet, and keep the one that grows the
running bounding box the least. If every candidate would overlap, cut this
boundary entirely and start the panel as a new, separate sticker instead of
forcing a collision -- trading one piece-penalty for feasibility.

This never explores every spanning tree (that is left as headroom), but it
consistently dodges the overlap trap and lands much closer to a compact
rectangle than the fixed-column greedy pass.
"""
import sys
from collections import deque

DX = [0, 1, 0, -1]
DY = [1, 0, -1, 0]
OPP = {(0, 2), (2, 0), (1, 3), (3, 1)}


def rot90cw(vec):
    dx, dy = vec
    return (dy, -dx)


def rot(r, vec):
    for _ in range(r % 4):
        vec = rot90cw(vec)
    return vec


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); m = int(toks[1])
    idx = 3
    edges = []
    for _ in range(m):
        u = int(toks[idx]); su = int(toks[idx + 1])
        v = int(toks[idx + 2]); sv = int(toks[idx + 3])
        idx += 4
        edges.append((u, su, v, sv))

    flat_idx, fold_idx = [], []
    for i, (u, su, v, sv) in enumerate(edges):
        (flat_idx if (su, sv) in OPP else fold_idx).append(i)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    kept_flat = []
    for i in flat_idx:
        u, su, v, sv = edges[i]
        if find(u) != find(v):
            union(u, v)
            kept_flat.append(i)

    panel_of = [find(x) for x in range(n)]
    panels = {}
    for x in range(n):
        panels.setdefault(panel_of[x], []).append(x)

    adj_flat = [[] for _ in range(n)]
    for i in kept_flat:
        u, su, v, sv = edges[i]
        adj_flat[u].append((v, su, sv))
        adj_flat[v].append((u, sv, su))

    local_pos = {}
    local_rot = {}
    visited = [False] * n
    for pid, nodes in panels.items():
        root = min(nodes)
        local_pos[root] = (0, 0)
        local_rot[root] = 0
        visited[root] = True
        stack = [root]
        while stack:
            u = stack.pop()
            for (v, su, sv) in adj_flat[u]:
                if visited[v]:
                    continue
                g = (su + local_rot[u]) % 4
                local_pos[v] = (local_pos[u][0] + DX[g], local_pos[u][1] + DY[g])
                local_rot[v] = (g + 2 - sv) % 4
                visited[v] = True
                stack.append(v)

    fold_groups = {}
    for i in fold_idx:
        u, su, v, sv = edges[i]
        pu, pv = panel_of[u], panel_of[v]
        if pu == pv:
            continue
        key = frozenset((pu, pv))
        fold_groups.setdefault(key, []).append(i)

    panel_adj = {}
    for key in fold_groups:
        a, b = tuple(key)
        panel_adj.setdefault(a, []).append(b)
        panel_adj.setdefault(b, []).append(a)

    placed_global = {}
    placed_cells_by_comp = {}
    comp_of_panel = {}
    kept_final = list(kept_flat)
    next_comp = [0]

    def place_identity(pid, comp_id):
        for u in panels[pid]:
            placed_global[u] = (local_pos[u][0], local_pos[u][1], local_rot[u])
        placed_cells_by_comp.setdefault(comp_id, set())
        for u in panels[pid]:
            placed_cells_by_comp[comp_id].add(placed_global[u][:2])
        comp_of_panel[pid] = comp_id

    root_panel = panel_of[0]
    place_identity(root_panel, next_comp[0])
    next_comp[0] += 1

    visited_panel = {root_panel}
    dq = deque([root_panel])
    while dq:
        pu = dq.popleft()
        for pv in panel_adj.get(pu, []):
            if pv in visited_panel:
                continue
            visited_panel.add(pv)
            dq.append(pv)
            key = frozenset((pu, pv))
            best = None
            for i in fold_groups[key]:
                a, sa, b, sb = edges[i]
                if panel_of[a] == pu:
                    U, su, V, sv = a, sa, b, sb
                else:
                    U, su, V, sv = b, sb, a, sa
                gx, gy, grot = placed_global[U]
                g = (su + grot) % 4
                Vg = (gx + DX[g], gy + DY[g])
                Vg_rot = (g + 2 - sv) % 4
                delta = (Vg_rot - local_rot[V]) % 4
                lx0, ly0 = local_pos[V]
                comp_id = comp_of_panel[pu]
                existing = placed_cells_by_comp.get(comp_id, set())
                newpos = {}
                seen_new = set()
                ok = True
                for w in panels[pv]:
                    lx, ly = local_pos[w]
                    rvec = rot(delta, (lx - lx0, ly - ly0))
                    p = (Vg[0] + rvec[0], Vg[1] + rvec[1])
                    if p in existing or p in seen_new:
                        ok = False
                        break
                    seen_new.add(p)
                    newpos[w] = (p[0], p[1], (local_rot[w] + delta) % 4)
                if not ok:
                    continue
                allpts = existing | seen_new
                xs = [p[0] for p in allpts]
                ys = [p[1] for p in allpts]
                area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
                if best is None or area < best[0]:
                    best = (area, i, newpos)
            if best is None:
                comp_id = next_comp[0]
                next_comp[0] += 1
                place_identity(pv, comp_id)
            else:
                _, i, newpos = best
                kept_final.append(i)
                comp_id = comp_of_panel[pu]
                comp_of_panel[pv] = comp_id
                for w, val in newpos.items():
                    placed_global[w] = val
                    placed_cells_by_comp[comp_id].add(val[:2])

    print(len(kept_final))
    print(" ".join(map(str, kept_final)))


if __name__ == "__main__":
    main()
