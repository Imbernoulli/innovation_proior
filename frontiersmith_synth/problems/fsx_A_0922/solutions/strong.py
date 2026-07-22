# TIER: strong
# Marginal-cost water-filling: within each round, insert packets one at a
# time (largest weight first, for stable bin-covering) onto whichever link
# currently has the LOWEST MARGINAL delay increase from adding that packet
# -- recomputed after every insertion, so the policy internalizes the load
# it is itself concurrently committing this round instead of freezing on a
# pre-round snapshot. This is the insight the trap punishes: no single
# "currently cheapest" link is chased blindly; traffic is spread so every
# used link's marginal cost stays balanced, damping the herding/ping-pong
# feedback the greedy tier falls into. Backlog is simulated round to round
# using the policy's own committed choices (the full plan is known up front).
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


def cost_at(e, x):
    return x * latency(e, x) if x > 0 else 0.0


backlog = [0.0] * K
routes = []
for rnd in rounds:
    n = rnd["n"]
    weights = rnd["weights"]
    order = sorted(range(n), key=lambda i: -weights[i])
    committed = list(backlog)
    row = [0] * n
    for idx in order:
        w = weights[idx]
        best_e, best_marg = 0, None
        for e in range(K):
            x0 = committed[e]
            marg = cost_at(edges[e], x0 + w) - cost_at(edges[e], x0)
            if best_marg is None or marg < best_marg - 1e-12:
                best_marg = marg
                best_e = e
        row[idx] = best_e
        committed[best_e] += w
    for e in range(K):
        backlog[e] = max(0.0, committed[e] - edges[e]["cap"]) * decay
    routes.append(row)

print(json.dumps({"routes": routes}))
