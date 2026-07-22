# TIER: greedy
# "Trust the neighbourhood you started in": look only at the start point and
# the short-hop local probes taken right around it, pick whichever reading is
# best, and drill a fixed-radius jittered cluster of wells around that single
# point. Never looks at the coarse scan_probes, so on instances where the
# tallest deposit sits elsewhere in the box (only visible via scan_probes) it
# is capped at the secondary deposit it started next to, however precisely it
# refines there.
import sys, json, random

inst = json.load(sys.stdin)
D = inst["dim"]
box = inst["box"]
Q = inst["budget"]
start = inst["start"]
start_v = inst["start_value"]

pool = [{"x": start, "value": start_v}] + inst["local_probes"]
best = max(pool, key=lambda p: p["value"])

STEP = 0.6           # fixed step, not adapted to anything
rnd = random.Random(20260714)
queries = []
for _ in range(Q):
    pt = [min(max(best["x"][d] + rnd.gauss(0.0, STEP), box[d][0]), box[d][1])
          for d in range(D)]
    queries.append(pt)

print(json.dumps({"queries": queries}))
