# TIER: strong
# The insight: route by VALUE DENSITY (expected value per slew-second), not raw
# value, and DEFER cloud-risky targets to the second pass instead of gambling on
# pass 1.
#
#   1) Triage: a target whose pass-1 cloud forecast is high (>=0.5) AND whose
#      pass-2 risk-adjusted, decay-discounted value clearly exceeds its pass-1
#      risk-adjusted value (by a safety margin) is assigned to the PASS-2 pool
#      instead of pass 1 -- it is worth eating the extra decay to dodge near-
#      certain cloud loss now. Everything else stays in the pass-1 pool.
#   2) Within each pool, build the visiting order greedily by "value per
#      slew-second": repeatedly extend the route with whichever remaining target
#      maximizes (forecast-risk-adjusted value) / (marginal slew+settle+dwell
#      cost from the route's current end), among targets that still fit the
#      remaining pass budget. This is the value-density routing insight -- it
#      naturally prefers a cluster of cheap, reliable targets over a single
#      expensive, uncertain "prize" that would burn the whole budget getting
#      there for one shot.
import sys, json

inst = json.load(sys.stdin)
slew_rate = inst["slew_rate"]
settle = inst["settle"]
pass_gap = inst["pass_gap"]
targets = inst["targets"]

DEFER_MARGIN = 1.15

pool1, pool2 = [], []
for t in targets:
    rv1 = t["value"] * (1.0 - t["cloud_forecast_p1"])
    decay2 = max(0.0, 1.0 - t["decay_rate"] * (pass_gap + inst["pass2_budget"] * 0.5))
    rv2 = t["value"] * (1.0 - t["cloud_forecast_p2"]) * decay2
    if t["cloud_forecast_p1"] >= 0.5 and rv2 > rv1 * DEFER_MARGIN:
        pool2.append(t)
    else:
        pool1.append(t)


def route_by_density(pool, budget, is_pass1):
    remaining = {t["id"]: t for t in pool}
    pos = 0.0
    elapsed = 0.0
    order = []
    while remaining:
        best_id = None
        best_density = -1.0
        best_cost = 0.0
        for tid, t in remaining.items():
            cost = abs(t["x"] - pos) * slew_rate + settle + t["dwell"]
            if elapsed + cost > budget:
                continue
            p = t["cloud_forecast_p1"] if is_pass1 else t["cloud_forecast_p2"]
            rv = t["value"] * (1.0 - p)
            density = rv / cost if cost > 1e-9 else 0.0
            if density > best_density:
                best_density = density
                best_id = tid
                best_cost = cost
        if best_id is None:
            break
        t = remaining.pop(best_id)
        elapsed += best_cost
        pos = t["x"]
        order.append(best_id)
    return order


order1 = route_by_density(pool1, inst["pass1_budget"], True)
order2 = route_by_density(pool2, inst["pass2_budget"], False)
print(json.dumps({"pass1": order1, "pass2": order2}))
