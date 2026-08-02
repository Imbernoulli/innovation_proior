# TIER: strong
# Insight: route by EXPECTED HARM x DECISION-UNCERTAINTY, not by raw model
# confidence, and ROTATE reviewers to keep effective accuracy up.
#
# For each item we estimate:
#   est_cost_auto  = expected cost of leaving it to the auto-rule, using
#                    model_score as our best P(violation) estimate (mirrors
#                    exactly how the evaluator would score an unreviewed item
#                    in expectation)
#   est_exp_loss   = model_score*severity + (1-model_score)*appeal_cost
#                    (expected magnitude of getting it wrong either way)
#   est_value      = est_cost_auto - (1 - fresh_best_acc) * est_exp_loss
# est_value is > 0 only when review is expected to help: items the auto-rule
# is already very likely to get right (extreme model_score, low severity --
# classic "confident spam") get NEGATIVE value and are deliberately skipped,
# even if a reviewer slot is free. Items with moderate model_score (auto-rule
# is basically guessing) and high severity/appeal weight get high value even
# though the model itself isn't "confident" about them.
#
# We then spend capacity on the highest-value items in ROUND-ROBIN order across
# reviewers sorted by base accuracy, so no single reviewer absorbs a long run of
# consecutive assignments (which would tank their effective accuracy via
# fatigue) -- this is the "rotate reviewers to hold accuracy up" half of the
# insight.
import sys, json

inst = json.load(sys.stdin)
items = inst["items"]
cats = inst["categories"]
reviewers = inst["reviewers"]
thr = inst["auto_threshold"]
floor_acc = inst["floor_acc"]
diff_pen = inst["difficulty_penalty"]

N = len(items)
R = len(reviewers)


def est_cost_auto(it):
    c = cats[it["category"]]
    s = it["model_score"]
    if s >= thr:
        return (1.0 - s) * c["appeal_cost"]
    return s * c["severity"]


def est_exp_loss(it):
    c = cats[it["category"]]
    s = it["model_score"]
    return s * c["severity"] + (1.0 - s) * c["appeal_cost"]


best_base_acc = max(rv["base_accuracy"] for rv in reviewers) if reviewers else floor_acc

value = []
for it in items:
    ea0 = max(floor_acc, best_base_acc - diff_pen * it["ambiguity"])
    v = est_cost_auto(it) - (1.0 - ea0) * est_exp_loss(it)
    value.append(v)

order = sorted(range(N), key=lambda i: -value[i])

rv_sorted = sorted(reviewers, key=lambda rv: -rv["base_accuracy"])
cap_left = {rv["id"]: rv["capacity"] for rv in reviewers}
schedule = {str(rv["id"]): [] for rv in reviewers}

total_cap = sum(rv["capacity"] for rv in reviewers)
ptr = 0
assigned = 0
for i in order:
    if value[i] <= 0:
        continue
    if assigned >= total_cap:
        break
    tries = 0
    while cap_left[rv_sorted[ptr % R]["id"]] <= 0 and tries < R:
        ptr += 1
        tries += 1
    rid = rv_sorted[ptr % R]["id"]
    if cap_left[rid] <= 0:
        continue
    schedule[str(rid)].append(i)
    cap_left[rid] -= 1
    ptr += 1
    assigned += 1

print(json.dumps({"schedule": schedule}))
