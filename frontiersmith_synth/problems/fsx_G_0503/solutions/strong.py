# TIER: strong
# Stratum-wise threshold search with shrinkage and unlabeled target-score adaptation.
# First fit a global calibration threshold, then fit each unit|acuity stratum, shrink
# thin strata toward the global threshold, and partially move each threshold to preserve
# the calibrated source alert tail on the recent unlabeled target stream.
import json
import math
import sys


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


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


def best_threshold(records, grid, costs, fallback):
    if not records:
        return fallback
    best_t = fallback
    best_u = None
    for t in grid:
        t = float(t)
        u = patient_utility(records, t, costs)
        if best_u is None or u > best_u + 1e-12:
            best_u = u
            best_t = t
    return best_t


def quantile_threshold(scores, tail_rate):
    if not scores:
        return None
    scores = sorted(scores)
    tail_rate = clip(tail_rate, 0.02, 0.80)
    idx = int(math.floor((1.0 - tail_rate) * (len(scores) - 1)))
    return scores[max(0, min(len(scores) - 1, idx))]


inst = json.load(sys.stdin)
grid = inst.get("threshold_grid") or [round(0.05 + 0.025 * i, 3) for i in range(31)]
cal = inst["calibration"]
recent = inst["recent_unlabeled"]
costs = inst["costs"]
baseline = float(inst.get("baseline_threshold", 0.50))
global_t = best_threshold(cal, grid, costs, baseline)

by_key = {k: [] for k in inst["strata"]}
recent_by_key = {k: [] for k in inst["strata"]}
for rec in cal:
    by_key[rec["unit"] + "|" + rec["acuity"]].append(rec)
for rec in recent:
    recent_by_key[rec["unit"] + "|" + rec["acuity"]].append(rec["score"])

all_recent_scores = [rec["score"] for rec in recent]
all_cal_scores = [rec["score"] for rec in cal]
table = {}
for key in inst["strata"]:
    rows = by_key[key]
    local = best_threshold(rows, grid, costs, global_t)
    alpha = min(0.78, max(0.25, len(rows) / 160.0))
    t = alpha * local + (1.0 - alpha) * global_t
    source_scores = [rec["score"] for rec in rows]
    if source_scores:
        tail = sum(1 for x in source_scores if x >= t) / len(source_scores)
    else:
        tail = sum(1 for x in all_cal_scores if x >= t) / max(1, len(all_cal_scores))
    target_scores = recent_by_key[key] or all_recent_scores
    qt = quantile_threshold(target_scores, tail)
    if qt is not None:
        t = 0.72 * t + 0.28 * qt
    table[key] = round(clip(t, 0.04, 0.92), 6)

print(json.dumps({"default_threshold": round(global_t, 6), "thresholds": table}))
