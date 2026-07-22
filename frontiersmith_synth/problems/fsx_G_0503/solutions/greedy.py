# TIER: greedy
# Fit one global threshold on the labeled calibration stream.  It respects the
# asymmetric miss cost and alert cost, but ignores unit/acuty heterogeneity and target
# distribution shift.
import json
import math
import sys


def patient_utility(records, threshold, costs):
    fn_cost = costs["fn_cost_by_acuity"]
    alert_cost = float(costs["alert_cost"])
    tp_credit = float(costs["tp_credit"])
    util = 0.0
    alerts = 0
    for rec in records:
        y = int(rec.get("label", 0))
        alert = rec["score"] >= threshold
        if alert:
            alerts += 1
        fc = float(fn_cost[rec["acuity"]])
        if y:
            util += tp_credit if alert else -fc
        elif alert:
            util -= alert_cost
        if alert:
            util -= alert_cost
    rate = alerts / max(1, len(records))
    if rate > 0.28:
        util -= 0.35 * len(records) * (rate - 0.28) * (rate - 0.28)
    return util


inst = json.load(sys.stdin)
grid = inst.get("threshold_grid") or [0.05 + 0.025 * i for i in range(31)]
cal = inst["calibration"]
costs = inst["costs"]
best_t = float(inst.get("baseline_threshold", 0.50))
best_u = None
for t in grid:
    t = float(t)
    u = patient_utility(cal, t, costs)
    if best_u is None or u > best_u + 1e-12:
        best_u = u
        best_t = t

print(json.dumps({"default_threshold": round(best_t, 6), "thresholds": {}}))
