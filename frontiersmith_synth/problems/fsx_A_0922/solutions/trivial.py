# TIER: trivial
# Static cheapest-link routing: pick the single link with the lowest base
# (zero-load) latency t0 and route EVERY packet, every round, through it.
# Completely ignores load, capacity, and the queue it builds up.
import sys, json

inst = json.load(sys.stdin)
edges = inst["edges"]
rounds = inst["rounds"]

best = min(range(len(edges)), key=lambda i: (edges[i]["t0"], i))

routes = [[best] * rnd["n"] for rnd in rounds]

print(json.dumps({"routes": routes}))
