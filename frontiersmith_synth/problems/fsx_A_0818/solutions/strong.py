# TIER: strong
# Innovation: spend the few structured local probes to ESTIMATE the field's
# correlation length (how much value changes per unit distance right around
# the start), then set BOTH the exploration step size and the restart
# aggressiveness from that estimate:
#   - low ruggedness (long correlation length) -> the field looks like one
#     smooth basin -> seed ONE tight, small-step cluster around the single
#     best pilot reading (fine local refinement).
#   - high ruggedness (short correlation length) -> the field is probably
#     multimodal -> don't trust the local neighbourhood alone: seed SEVERAL
#     elite-restart clusters from the best pilot readings ACROSS THE WHOLE
#     BOX (including the coarse scan_probes, which is how a taller, distant
#     deposit gets found), with a wider step size per cluster.
import sys, json, random

inst = json.load(sys.stdin)
D = inst["dim"]
box = inst["box"]
Q = inst["budget"]
start = inst["start"]
start_v = inst["start_value"]
local = inst["local_probes"]
scan = inst["scan_probes"]

# --- ruggedness-correlation-probe: per-radius slope of |value - start_value| ---
slopes = [abs(p["value"] - start_v) / max(p["r"], 1e-6) for p in local]
allvals = [start_v] + [p["value"] for p in local] + [p["value"] for p in scan]
scale = max(1e-9, max(allvals) - min(allvals))
rug_ratio = (max(slopes) - min(slopes)) / scale if slopes else 0.0

RUGGED = rug_ratio > 0.35

pool = [{"x": start, "value": start_v}] + local + scan
pool.sort(key=lambda p: p["value"], reverse=True)

if RUGGED:
    n_elites = min(4, len(pool))
    step = 0.9
else:
    n_elites = 1
    step = 0.22

elites = pool[:n_elites]
rnd = random.Random(20260714)

queries = []
per = max(1, Q // n_elites)
for e in elites:
    for _ in range(per):
        if len(queries) >= Q:
            break
        pt = [min(max(e["x"][d] + rnd.gauss(0.0, step), box[d][0]), box[d][1])
              for d in range(D)]
        queries.append(pt)

# top up any remainder with finer jitter around the single best elite
best_e = elites[0]
while len(queries) < Q:
    pt = [min(max(best_e["x"][d] + rnd.gauss(0.0, step * 0.4), box[d][0]), box[d][1])
          for d in range(D)]
    queries.append(pt)

queries = queries[:Q]
print(json.dumps({"queries": queries}))
