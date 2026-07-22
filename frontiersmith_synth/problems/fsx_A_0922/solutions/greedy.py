# TIER: greedy
# Myopic least-latency routing: each round, look at the per-link state as of
# the START of the round (the backlog carried in from before -- a frozen
# pre-round snapshot), pick the SINGLE link that currently looks cheapest,
# and route the ENTIRE round's packets through it. Never accounts for the
# load it is itself concurrently piling on -- the classic "just pick
# whatever looks fastest right now" instinct. Simulates its own backlog
# trajectory round by round (it has the full plan up front) so its choices
# are self-consistent, but the per-round DECISION never looks past the
# single frozen snapshot.
import sys, json

inst = json.load(sys.stdin)
edges = inst["edges"]
decay = inst["decay"]
rounds = inst["rounds"]
K = len(edges)


def latency(e, x):
    if x <= 0:
        return e["t0"]
    return e["t0"] * (1.0 + (x / e["cap"]) ** e["p"])


backlog = [0.0] * K
routes = []
for rnd in rounds:
    best = min(range(K), key=lambda e: (latency(edges[e], backlog[e]), e))
    row = [best] * rnd["n"]

    # advance the real dynamics using this round's actual (bad) choice, so
    # next round's snapshot reflects what really happened
    load = list(backlog)
    for w, e in zip(rnd["weights"], row):
        load[e] += w
    for e in range(K):
        backlog[e] = max(0.0, load[e] - edges[e]["cap"]) * decay

    routes.append(row)

print(json.dumps({"routes": routes}))
