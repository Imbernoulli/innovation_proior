# TIER: strong
# Probe-then-commit with trend extrapolation, not magnitude extrapolation.
#
# For each solver, look at the THREE successive increments between its four
# probe checkpoints (d1, d2, d3). The ratio d3/d2 tells you the SHAPE of the
# curve just past the probe window, which is far more informative than the
# raw current value:
#   - ratio > 1.05: increments are GROWING -> the curve is still accelerating
#     (a possible "sleeper" that has not taken off yet). Extrapolate
#     optimistically: project the compounding growth forward.
#   - ratio <= 1.05 (including a flat trace): increments are shrinking or
#     dead -> the curve is already saturating. Extrapolate its remaining
#     geometric tail, which stays close to the current value.
# Rank solvers by this extrapolated estimate (an informativeness-probe read of
# "who will still be climbing"), then split the budget across the top few
# ranked solvers with sharply decaying weights: the top pick gets the
# majority (so a correct read pays off almost like full commitment), but
# every other candidate still gets a nonzero reserve, so a wrong read about
# which solver is truly accelerating doesn't zero out the runner-up -- the
# "revisit if it plateaus" insurance a single blind commitment can't offer.
import sys, json

inst = json.load(sys.stdin)
heuristics = inst["heuristics"]
budget = inst["budget"]

TOPW = [0.42, 0.26, 0.14, 0.09, 0.06, 0.03]


def extrapolate(probe):
    p0, p1, p2, p3 = probe
    d2 = p2 - p1
    d3 = p3 - p2
    if d3 < 1e-7 and d2 < 1e-7:
        return p3 * 1.05
    ratio = d3 / max(d2, 1e-9)
    if ratio > 1.05:
        est = p3 + d3 * min(ratio, 3.0) * 6.0
        return min(est, 1.3)
    r = min(ratio, 0.97)
    tail = d3 * r / max(1.0 - r, 1e-3)
    return p3 + tail


scored = sorted(heuristics, key=lambda h: -extrapolate(h["probe"]))

alloc = {}
spent = 0
n = len(scored)
for rank, h in enumerate(scored):
    w = TOPW[rank] if rank < len(TOPW) else 0.0
    amt = int(round(budget * w))
    alloc[h["id"]] = amt
    spent += amt

# fix rounding so the total lands exactly on/under budget
diff = spent - budget
if diff > 0 and scored:
    tail_id = scored[-1]["id"]
    alloc[tail_id] = max(0, alloc[tail_id] - diff)
elif diff < 0 and scored:
    top_id = scored[0]["id"]
    alloc[top_id] += (-diff)

print(json.dumps({"alloc": alloc}))
