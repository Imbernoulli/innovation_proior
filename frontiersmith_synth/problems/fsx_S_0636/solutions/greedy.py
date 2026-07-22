# TIER: greedy
# Textbook DSATUR (saturation-degree-order) register-allocation coloring: always
# color next whichever uncolored node has the most DISTINCT colors already forced
# among its colored neighbors ("saturation degree"), breaking ties by static
# degree then by index; assign it the smallest available color, or spill it if
# none is free. This is the standard "obvious approach an average coder writes
# first" for interference-graph coloring -- but it is completely BLIND to spill
# weight, so on instances where a dense cluster of CHEAP nodes reaches maximum
# saturation before an EXPENSIVE node does, it happily exhausts every color on
# the cheap cluster and strands the expensive node with nothing left, even
# though sacrificing one cheap node instead would have been far better.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; k = inst["k"]; edges = inst["edges"]

adj = [set() for _ in range(n)]
for (u, v) in edges:
    adj[u].add(v)
    adj[v].add(u)
deg = [len(adj[i]) for i in range(n)]

colors = [None] * n
colored_neighbor_colors = [set() for _ in range(n)]
remaining = set(range(n))

while remaining:
    # pick node maximizing (saturation degree, static degree, -index) [asc index]
    best = None
    best_key = None
    for u in remaining:
        key = (len(colored_neighbor_colors[u]), deg[u], -u)
        if best_key is None or key > best_key:
            best_key = key
            best = u
    used = colored_neighbor_colors[best]
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
                colored_neighbor_colors[w].add(c)

print(json.dumps({"colors": [(-1 if c is None else c) for c in colors]}))
