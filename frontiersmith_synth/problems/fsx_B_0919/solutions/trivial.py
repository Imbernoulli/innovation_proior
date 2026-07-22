# TIER: trivial
# Blind single-node serial: run every job on node 0, the instant it is
# legally allowed to start, in arrival order. Ignores the other 8 nodes
# and the heat model entirely (baseline "do the obvious dumbest thing").
import sys, json

def main():
    inst = json.load(sys.stdin)
    jobs = sorted(inst["jobs"], key=lambda j: (j["arrival"], j["id"]))
    T = inst["T"]
    node_free = 0
    sched = []
    for j in jobs:
        start = max(j["arrival"], node_free)
        if start + j["demand"] > T:
            sched.append({"id": j["id"], "node": -1, "start": 0})
            continue
        node_free = start + j["demand"]
        sched.append({"id": j["id"], "node": 0, "start": start})
    print(json.dumps({"schedule": sched}))

main()
