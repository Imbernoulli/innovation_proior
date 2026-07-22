# TIER: greedy
# "Coolest node now" -- the obvious first instinct: keep a private,
# self-only heat proxy per node and always drop the next job on whichever
# node the proxy currently reads coolest, starting immediately. It uses a
# generic, FIXED fast-decay assumption (a plausible-looking constant) for
# how quickly a node cools instead of reading the instance's OWN (often
# much slower) decay -- and it never reads alpha/HI/LO at all, so it has
# no model of grid-neighbor diffusion or of hysteresis (that recovery
# needs cooling all the way to LO, not just "seems idle for a bit"). The
# result: it reuses a node the moment its crude proxy looks low, even
# though the node's TRUE temperature (driven by the real, slower decay,
# plus heat still arriving from hot neighbors) is still elevated -- and it
# never delays a job to let a hot spot dissipate.
import sys, json

ASSUMED_DECAY = 0.5  # generic fast-cooling assumption; ignores the real (slower) `decay`

def main():
    inst = json.load(sys.stdin)
    N, T = inst["N"], inst["T"]
    jobs = sorted(inst["jobs"], key=lambda j: (j["arrival"], j["id"]))

    proxy = [0.0] * N
    last_t = [0] * N
    node_free = [0] * N
    sched = []

    for j in jobs:
        now = j["arrival"]
        for i in range(N):
            dt = max(0, now - last_t[i])
            proxy[i] *= (1.0 - ASSUMED_DECAY) ** dt
            last_t[i] = now

        node = min(range(N), key=lambda i: proxy[i])
        start = max(j["arrival"], node_free[node])
        if start + j["demand"] > T:
            sched.append({"id": j["id"], "node": -1, "start": 0})
            continue
        node_free[node] = start + j["demand"]
        proxy[node] += j["heat_rate"] * j["demand"]
        sched.append({"id": j["id"], "node": node, "start": start})

    print(json.dumps({"schedule": sched}))

main()
