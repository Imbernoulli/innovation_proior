# TIER: strong
# Weight-aware saturation-degree order + Kempe-chain repair.
#
#   1) Build the coloring like DSATUR (always color the currently most-saturated
#      node next), but break ties by spill WEIGHT (highest first), not just static
#      degree/index. Since every node starts at saturation 0, this means a very
#      expensive node gets seated with a free color BEFORE a cheap dense cluster
#      has a chance to saturate around it and take every color for itself.
#   2) For any node that still ends up spilled (a genuine structural bottleneck,
#      e.g. inside a tight clique), try to rescue it: for each color c blocked by
#      exactly one neighbor u, look for a KEMPE CHAIN -- the connected component
#      reachable from u using only nodes currently colored c or some other color
#      d. Flipping c<->d on that whole component is ALWAYS safe (every edge
#      inside it already alternates c/d in a proper coloring, so swapping both
#      colors keeps every edge proper), and if after flipping none of the spilled
#      node's neighbors hold c any more, color c is free -- rescue it with NO new
#      spill anywhere. If no such chain exists but a lower-weight neighbor blocks
#      every option, evict that cheaper neighbor instead (only when it strictly
#      lowers total spill weight) and try to re-seat the evicted neighbor with a
#      leftover color. Repeat a few rounds since a rescue can open the door for
#      the next one.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; k = inst["k"]; weights = inst["weights"]; edges = inst["edges"]

adj = [set() for _ in range(n)]
for (u, v) in edges:
    adj[u].add(v)
    adj[v].add(u)
deg = [len(adj[i]) for i in range(n)]


def build_weight_biased_dsatur():
    colors = [None] * n
    seen_colors = [set() for _ in range(n)]
    remaining = set(range(n))
    while remaining:
        best = None
        best_key = None
        for u in remaining:
            key = (len(seen_colors[u]), weights[u], deg[u], -u)
            if best_key is None or key > best_key:
                best_key = key
                best = u
        used = seen_colors[best]
        c = None
        for cand in range(k):
            if cand not in used:
                c = cand
                break
        colors[best] = c if c is not None else -1
        remaining.discard(best)
        if c is not None:
            for w in adj[best]:
                if w in remaining:
                    seen_colors[w].add(c)
    return [(-1 if c is None else c) for c in colors]


def kempe_component(start, c, d, colors):
    comp = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for w in adj[node]:
            if w not in comp and colors[w] in (c, d):
                comp.add(w)
                stack.append(w)
    return comp


def repair_round(colors):
    """One pass: try to rescue every currently-spilled node, priciest first."""
    changed = False
    spilled = [v for v in range(n) if colors[v] == -1]
    spilled.sort(key=lambda v: -weights[v])
    for v in spilled:
        if colors[v] != -1:
            continue
        for c in range(k):
            blockers = [u for u in adj[v] if colors[u] == c]
            if not blockers:
                colors[v] = c
                changed = True
                break
            if len(blockers) != 1:
                continue
            u = blockers[0]
            rescued = False
            for d in range(k):
                if d == c:
                    continue
                comp = kempe_component(u, c, d, colors)
                new_colors = {node: (d if colors[node] == c else c) for node in comp}
                ok = True
                for w in adj[v]:
                    cw = new_colors.get(w, colors[w])
                    if cw == c:
                        ok = False
                        break
                if ok:
                    for node, cc in new_colors.items():
                        colors[node] = cc
                    colors[v] = c
                    rescued = True
                    changed = True
                    break
            if rescued:
                break
            if weights[v] > weights[u]:
                colors[u] = -1
                colors[v] = c
                used_u = set(colors[w] for w in adj[u] if colors[w] != -1)
                cu = None
                for cand in range(k):
                    if cand not in used_u:
                        cu = cand
                        break
                colors[u] = cu if cu is not None else -1
                changed = True
                break
    return changed


colors = build_weight_biased_dsatur()
for _ in range(4):
    if not repair_round(colors):
        break

print(json.dumps({"colors": colors}))
