# TIER: trivial
# Raw index-order first-fit: process nodes 0..n-1 in the order they were given,
# each taking the smallest color not yet used by an already-colored neighbor
# (spilling if none is free). Completely ignores weights -- this reproduces the
# evaluator's own weak baseline exactly, so it always scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; k = inst["k"]; edges = inst["edges"]

adj = [set() for _ in range(n)]
for (u, v) in edges:
    adj[u].add(v)
    adj[v].add(u)

colors = [-1] * n
for u in range(n):
    used = set(colors[w] for w in adj[u] if colors[w] != -1)
    c = None
    for cand in range(k):
        if cand not in used:
            c = cand
            break
    colors[u] = c if c is not None else -1

print(json.dumps({"colors": colors}))
